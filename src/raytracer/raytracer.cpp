/*
 * raytracer.cu
 *
 *  Created on: 4 Sep 2013
 *      Author: drw
 */

#include "raytracer.h"
#include "mino_stepper.h"
#include "ray_destination.h"
#include <cassert>

template <typename T>
Raytracer<T>::Raytracer( int num_rays, T spin_par, T init_precision, T init_max_phistep, T init_max_tstep )
	: nRays( num_rays )
	,  spin(spin_par)
	, precision(init_precision)
    , theta_precision(THETA_PRECISION)
	, max_phistep(init_max_phistep)
	, max_tstep(init_max_tstep)
    , maxtstep_rlim(MAXDT_RLIM)
    , rk45_tol(T(1e-8))
    , symp_step(T(0))
    , symp_order(6)
{
	//
	// Constructor function - allocates host and device memory for each ray to store ray position, momentum,
	// integration steps taken, redshift and constants of motion. Allocates only revice memory for the sign
	// of rdot and thetadot as well as the emitted energy used for redshift calculations.
	//
	// Rays are set up on a 2D grid of GPU threads (for easy variation of 2 parameters between rays)
	//
	// Arguments:
	//	num_rays	int		Number of rays required
	//	spin_par	T		Dimensionless spin parameter of black hole
	//	toler		T		Tolerance used to set step size in numerical integration of geodesics
	//

	// calculate horizon and store in GPU shared memory
	horizon = kerr_horizon<T>(spin);
	cout << "Event horizon at " << horizon << endl;

    int mb = 1<<20;

	cout << "Allocating memory (" << nRays*sizeof(Ray<T>) / mb << "MB)" << endl;
    rays = new Ray<T>[nRays];

    for(int ray=0; ray<nRays; ray++)
    {
	    rays[ray].steps = -1;
	    rays[ray].status = 0;
    }
}

template <typename T>
Raytracer<T>::~Raytracer( )
{
	//
	// Destructor - frees host and device memory used for ray tracing variables
	//
	cout << "Cleaning up raytracer" << endl;

	delete[] rays;
}

template <typename T>
void Raytracer<T>::run_raytrace(Integrator method, T theta_max, T r_max,
                                int show_progress, TextOutput* outfile,
                                int write_step, T write_rmax, T write_rmin, bool write_cartesian,
                                int steplim)
{
    //
    // Unified ray tracing entry point.  Selects the integration algorithm via the method argument:
    //   Integrator::Euler — fixed-step Euler integrator (default)
    //   Integrator::RK4   — classical 4th-order Runge-Kutta
    //   Integrator::RK45  — adaptive Dormand-Prince RK45/DOPRI5
    //   Integrator::Symplectic — Mino-time symplectic integrator (Störmer-Verlet / Yoshida)
    //
    static const char* names[] = {
        "Running raytracer...",
        "Running raytracer (RK4)...",
        "Running raytracer (RK45/DOPRI5)...",
        "Running raytracer (symplectic)..."
    };
    const int steplim = (method == Integrator::RK45) ? RK45_STEPLIM
                      : (method == Integrator::Symplectic) ? SYMP_STEPLIM : STEPLIM;
    cout << names[static_cast<int>(method)] << endl;

    ProgressBar prog(nRays, "Ray", 0, (show_progress > 0));
    show_progress = abs(show_progress);

    if (outfile != nullptr) {
        // Serial path: ordered file writes required
        for (int ray = 0; ray < nRays; ray++)
        {
            if (show_progress != 0 && (ray % show_progress) == 0) prog.show(ray + 1);
            if (rays[ray].steps < 0) continue;
            else if (rays[ray].steps >= effective_steplim) continue;

            switch (method) {
                case Integrator::Euler: propagate    (ray, r_max, theta_max, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::RK4:  propagate_rk4 (ray, r_max, theta_max, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::RK45: propagate_rk45(ray, r_max, theta_max, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::Symplectic: propagate_symplectic(ray, r_max, theta_max, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
            }
            outfile->newline(2);
        }
    } else {
        // Parallel path: rays are fully independent, no file I/O
        int rays_done = 0;
        #pragma omp parallel for schedule(dynamic)
        for (int ray = 0; ray < nRays; ray++)
        {
            if (show_progress != 0) {
                int done;
                #pragma omp atomic capture
                done = ++rays_done;
                if (done % show_progress == 0) {
                    #pragma omp critical
                    prog.show(done);
                }
            }
            if (rays[ray].steps < 0) continue;
            else if (rays[ray].steps >= effective_steplim) continue;

            switch (method) {
                case Integrator::Euler: propagate    (ray, r_max, theta_max, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::RK4:  propagate_rk4 (ray, r_max, theta_max, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::RK45: propagate_rk45(ray, r_max, theta_max, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::Symplectic: propagate_symplectic(ray, r_max, theta_max, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
            }
        }
    }
    prog.done();
}

template <typename T>
inline int Raytracer<T>::propagate(int ray, const T rlim, const T thetalim, const int steplim, TextOutput* outfile, int write_step, T write_rmax, T write_rmin, bool write_cartesian )
{
	//
	// propagate the photon along its geodesic until limiting r or theta reached
	//
	int steps = 0;

	bool r_was_positive     = false;  // only allow r sign flip after rdotsq has been positive
	bool theta_was_positive = true;   // allow flip on first step if initial direction is wrong

	T rhosq, delta;
	T rdotsq, thetadotsq;

	T step;

	T x, y, z;

	// copy variables locally to simplify the code
	T a = spin;
	T t = rays[ray].t;
	T r = rays[ray].r;
	T theta = rays[ray].theta;
	T phi = rays[ray].phi;
	T pt = rays[ray].pt;
	T pr = rays[ray].pr;
	T ptheta = rays[ray].ptheta;
	T pphi = rays[ray].pphi;
	int rdot_sign = rays[ray].rdot_sign;
	int thetadot_sign = rays[ray].thetadot_sign;
	int rdot_flips = rays[ray].rdot_flips;
	int equatorial_crossings = rays[ray].equatorial_crossings;
	T theta_eq_prev = theta;

	const T k = rays[ray].k;
	const T h = rays[ray].h;
	const T Q = rays[ray].Q;

	bool write_started = false;


	// integrate geodesic equations until limit reached
	// if thetalim is positive, we go until theta exceeds it, if it is negative, we go until it is less than the abs value to allow tracing back to theta=0
	while( r < rlim  && ( (thetalim > 0 && theta < thetalim) || (thetalim < 0 && theta > abs(thetalim)) || thetalim == 0 )  &&  steps < steplim )
	//while( r < rlim  && theta < thetalim  &&  steps < steplim )
	{
		++steps;

		const T sin_theta = sin(theta);
		const T cos_theta = cos(theta);
		const T sin2theta = sin_theta * sin_theta;
		rhosq = r*r + (a*cos_theta)*(a*cos_theta);
		delta = r*r - 2*r + a*a;
		const T rhosq_delta = rhosq * delta;

		// tdot
		pt = (rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h;
		pt /= rhosq_delta;

		// phidot
		pphi = 2*a*r*sin2theta*k + (rhosq - 2*r)*h;
		pphi /= sin2theta * rhosq_delta;

		// thetadot
		thetadotsq = Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta);
		thetadotsq = thetadotsq / (rhosq*rhosq);

		if (thetadotsq < 0 && theta_was_positive)
		{
			thetadot_sign *= -1;
			theta_was_positive = false;
			continue;
		}
		if (thetadotsq >= 0) theta_was_positive = true;

		// take the square roots and get the right signs
		ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

		// rdot
		rdotsq = k*pt - h*pphi - rhosq*ptheta*ptheta;
		rdotsq = rdotsq * delta/rhosq;

		if (rdotsq <= 0 && r_was_positive)
		{
			rdot_sign *= -1;
			r_was_positive = false;
			rdot_flips++;
		}
		else if (rdotsq > 0)
		{
			r_was_positive = true;
		}

		pr = sqrt(abs(rdotsq)) * rdot_sign;

        step = abs((r - (T) horizon) / pr) / precision;
        if(step > abs(theta / ptheta) / precision)
        {
            step = abs(theta / ptheta) / theta_precision;
        }
        if(max_tstep > 0 && r < maxtstep_rlim && step > abs(max_tstep / pt))
        {
            step = abs(max_tstep / pt);
        }
        if(max_phistep > 0 && step > abs(max_phistep / pphi))
        {
            step = abs(max_phistep / pphi);
        }
        // don't let the step be stupidly small
        if(step < MIN_STEP) step = MIN_STEP;

        // make sure we don't go past rlim
        if(rlim > 0 && r + pr * step > rlim) step = abs((rlim - r) / pr);
        // same for thetalim (but only if in range of r that would hit disc)
        if(thetalim > 0 && theta + ptheta * step > thetalim) step = abs((thetalim - theta) / ptheta);

        // the code below is if we wanted to be able to trace rays below the disc (not implemented)
//        if((!under) && thetalim > 0 && theta + ptheta * step > thetalim)
//        {
//            // would the new step size put us in the range of r for the disc?
//            if(r_disc <= 0 || ((r + pr * abs((thetalim - theta) / ptheta)) < r_disc &&
//                               (r + pr * abs((thetalim - theta) / ptheta)) > r_in))
//                step = abs((thetalim - theta) / ptheta);
//        }
//        else if(r_disc > 0 && under && thetalim > 0 && theta + ptheta * step < (M_PI - thetalim))
//        {
//            if((r + pr * abs((thetalim - theta) / ptheta)) < r_disc &&
//               (r + pr * abs((thetalim - theta) / ptheta)) > r_in)
//                step = abs((thetalim - theta) / ptheta);
//        }

        // the code below is if we wanted to run rays backwards (not implemented)
        //if(reverse) step *= -1;

		// flag the ray if tdot goes negative (inside the ergosphere - these are not physical)
		if(pt <= 0)
		{
			rays[ray].status |= RAY_STATUS_ERGO;
		}

		// check that the dot product of the photon 4-momentum with the local timelike Killing vector is positive
		if((1 - 2*r/rhosq) * pt + (2*a*r*sin2theta/rhosq) * pphi < 0)
		{
			rays[ray].status |= RAY_STATUS_NEG_ENERGY;
		}

		// calculate new position
		t += pt*step;
		r += pr*step;
		{ T tep = theta; theta += ptheta*step;
		  if ((tep < M_PI_2 && theta >= M_PI_2) || (tep > M_PI_2 && theta <= M_PI_2)) ++equatorial_crossings; }
		phi += pphi*step;
		// Reflect at polar axes: clamp theta to [0, pi]
		if (theta < T(0))    { theta = -theta;                thetadot_sign = -thetadot_sign; phi += T(M_PI); }
		if (theta > T(M_PI)) { theta = T(2)*T(M_PI) - theta; thetadot_sign = -thetadot_sign; phi += T(M_PI); }

		//aff += step;

		if(r <= horizon)
		{
			rays[ray].status |= RAY_STATUS_HORIZON;
			break;
		}

		if(outfile != 0 && (steps % write_step) == 0 )
		{
			if((write_rmax < 0 || r < write_rmax) && (write_rmin < 0 || r > write_rmin) )
			{
				write_started = true;
				if(write_cartesian)
				{
                    cartesian<T>(x, y, z, r, theta, phi, a);
					(*outfile) << t << x << y << z << endl;
				}
				else
				{
					(*outfile) << t << r << theta << phi << endl;
				}
			}
			else if(write_started)
			{
				break;
			}
		}
	}

	if (steps >= steplim)
		rays[ray].status |= RAY_STATUS_STEPLIM;
	else if (r >= rlim)
		rays[ray].status |= RAY_STATUS_RLIM;
	else if ((thetalim > 0 && theta >= thetalim) || (thetalim < 0 && theta <= abs(thetalim)))
		rays[ray].status |= RAY_STATUS_DEST;

	rays[ray].t = t;
    rays[ray].r = r;
    rays[ray].theta = theta;
    rays[ray].phi = phi;
    rays[ray].pt = pt;
    rays[ray].pr = pr;
    rays[ray].ptheta = ptheta;
    rays[ray].pphi = pphi;
    rays[ray].rdot_sign = rdot_sign;
    rays[ray].thetadot_sign = thetadot_sign;
    rays[ray].rdot_flips = rdot_flips;
    rays[ray].equatorial_crossings = equatorial_crossings;

	if(steps > 0) rays[ray].steps += steps;
	if (rays[ray].status & RAY_STATUS_STEPLIM)
		rays[ray].steps = -rays[ray].steps;   // negative steps flags a stuck/failed ray

	return steps;
}

template <typename T>
void Raytracer<T>::redshift_start(T V, bool reverse, bool projradius )
{
	//
	// Calculates the initial energy of each ray at emission for use in redshift calculations.
	// Emitter is orbitting the black hole rotation axis at angular velocity V, or if V = -1, the
	// angular velocity is calculated for a circular orbit at the current r co-ordinate of the ray.
	//
	// If ray is being propagated backwards in time (for image planes), the spin parameter is revered
	// (back to its true value as it will already have been reversed for the propagation) and the
	// spatial components of the ray's 4-momentum will be reversed so the ray is travelling in the
	// correct direction wrt the emitting material.
	//
	// This function should be called prior to running the raytrace if redshifts are required.
	//
	// Arguments:
	//	V		T		Angular velocity of emitter (set to -1 for circular orbit at current r of ray)
	//	reverse	bool	Whether the ray is being propagated backwards in time (for image planes, default value = false)
	//
	cout << "Calculating initial energies" << endl;

	for(int ray=0; ray<nRays; ray++)
	{
		T p[4];

		const T a = (reverse) ? -1*spin : spin;

		// metric coefficients
		const T rhosq = rays[ray].r*rays[ray].r + (a*cos(rays[ray].theta))*(a*cos(rays[ray].theta));
		const T delta = rays[ray].r*rays[ray].r - 2*rays[ray].r + a*a;
		const T sigmasq = (rays[ray].r*rays[ray].r + a*a)*(rays[ray].r*rays[ray].r + a*a) - a*a*delta*sin(rays[ray].theta)*sin(rays[ray].theta);

		const T e2nu = rhosq * delta / sigmasq;
		const T e2psi = sigmasq * sin(rays[ray].theta)*sin(rays[ray].theta) / rhosq;
		const T omega = 2*a*rays[ray].r / sigmasq;

		T g[16];
		for(int i=0; i<16; i++)
			g[i] = 0;

		// g[i][j] -> g[i*4 + j]
		g[0*4 + 0] = e2nu - omega*omega*e2psi;
		g[0*4 + 3] = omega*e2psi;
		g[3*4 + 0] = g[0*4 + 3];
		g[1*4 + 1] = -rhosq/delta;
		g[2*4 + 2] = -rhosq;
		g[3*4 + 3] = -e2psi;

		// if V==-1, calculate orbital velocity for a geodesic circular orbit in equatorial plane
		if(V == -1 && projradius)
			V = 1 / (a + rays[ray].r*sin(rays[ray].theta)*sqrt(rays[ray].r*sin(rays[ray].theta)));	// project the radius parallel to the equatorial plane
		else if(V == -1)
			V = 1 / (a + rays[ray].r*sqrt(rays[ray].r));

		// if(reverse) V *= -1;


		// timelike basis vector
		const T et[] = { (1/sqrt(e2nu))/sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu)
							, 0 , 0 ,
							(1/sqrt(e2nu))*V / sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu) };

		// photon momentum
        momentum_from_consts<T>(p[0], p[1], p[2], p[3], rays[ray].k, rays[ray].h, rays[ray].Q, rays[ray].rdot_sign,
                                rays[ray].thetadot_sign, rays[ray].r, rays[ray].theta, rays[ray].phi, spin);

		// if we're propagating backwards, reverse the direction of the photon momentum
		if(reverse) { p[1] *= -1; p[2] *= -1; p[3] *= -1; }

		// evaluate dot product to get energy
		rays[ray].emit = 0;
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
				rays[ray].emit += g[i*4 + j] * et[i]* p[j];
	}
}


template <typename T>
void Raytracer<T>::redshift(T V, bool reverse, bool projradius, int motion )
{
	//
	// Calculates the redshift of the ray (emitted / received energy).
	// Receiver is orbitting the black hole rotation axis at angular velocity V, or if V = -1, the
	// angular velocity is calculated for a circular orbit at the current r co-ordinate of the ray.
	//
	// If ray is being propagated backwards in time (for image planes), the spin parameter is revered
	// (back to its true value as it will already have been reversed for the propagation) and the
	// spatial components of the ray's 4-momentum will be reversed so the ray is travelling in the
	// correct direction wrt the receiving material.
	//
	// redshift_start( ) needs to have been called before rays were traced. Use this function after
	// propagation.
	//
	// Arguments:
	//	V		T		Angular velocity of receiver (set to -1 for circular orbit at current r of ray)
	//	reverse	bool	Whether the ray is being propagated backwards in time (for image planes, default value = false)
	//	projradius bool	Whether to use r*sin(theta) as the radial co-ordinate in angular velocity calculation to use radius projected parallel to equatorial plane
	//
	cout << "Calculating ray redshifts..." << endl;

	for(int ray=0; ray<nRays; ray++)
	{
		rays[ray].redshift = ray_redshift(V, reverse, projradius, rays[ray].r, rays[ray].theta, rays[ray].phi, rays[ray].k, rays[ray].h, rays[ray].Q, rays[ray].rdot_sign, rays[ray].thetadot_sign, rays[ray].emit, motion);
	}
}


template <typename T>
void Raytracer<T>::redshift(RayDestination<T>* dest, bool reverse, bool projradius, int motion)
{
	//
	// Calculates the redshift of each ray using a full 4-velocity field supplied by the
	// RayDestination object. For each ray, dest->four_velocity(r, theta, phi, spin, et) is
	// called to obtain the observer 4-velocity et[4] at the ray's termination point. The
	// GR dot product g_μν et^μ p^ν is then evaluated to compute the received energy.
	//
	// Arguments:
	//   dest        RayDestination<T>*  Destination object supplying the 4-velocity field
	//   reverse     bool                Whether the ray is being propagated backwards in time
	//   projradius  bool                Unused (kept for API compatibility)
	//   motion      int                 Unused (kept for API compatibility; 4-vector subsumes both modes)
	//
	cout << "Calculating ray redshifts..." << endl;

	for(int ray=0; ray<nRays; ray++)
	{
		T et[4];
		dest->four_velocity(rays[ray].r, rays[ray].theta, rays[ray].phi, spin, et);
		rays[ray].redshift = ray_redshift(et, reverse,
		                                  rays[ray].r, rays[ray].theta, rays[ray].phi,
		                                  rays[ray].k, rays[ray].h, rays[ray].Q,
		                                  rays[ray].rdot_sign, rays[ray].thetadot_sign,
		                                  rays[ray].emit);
	}
}


template <typename T>
inline T Raytracer<T>::ray_redshift( T V, bool reverse, bool projradius, T r, T theta, T phi, T k, T h, T Q, int rdot_sign, int thetadot_sign, T emit, int motion )
{
	// calculate the redshift of a single ray

	T p[4];

	// if propagating backwards (image plane), flip spin back to physical value
	const T a = (reverse) ? -1*spin : spin;

	// metric coefficients
	const T rhosq = r*r + (a*cos(theta))*(a*cos(theta));
	const T delta = r*r - 2*r + a*a;
	const T sigmasq = (r*r + a*a)*(r*r + a*a) - a*a*delta*sin(theta)*sin(theta);

	const T e2nu = rhosq * delta / sigmasq;
	const T e2psi = sigmasq * sin(theta)*sin(theta) / rhosq;
	const T omega = 2*a*r / sigmasq;

	T g[16];
	for(int i=0; i<16; i++)
		g[i] = 0;

	// g[i][j] -> g[i*4 + j]
	g[0*4 + 0] = e2nu - omega*omega*e2psi;
	g[0*4 + 3] = omega*e2psi;
	g[3*4 + 0] = g[0*4 + 3];
	g[1*4 + 1] = -rhosq/delta;
	g[2*4 + 2] = -rhosq;
	g[3*4 + 3] = -e2psi;

	T et[] = {0, 0, 0, 0};

	if(motion == 0)
	{
		// if V==-1, calculate orbital velocity for a geodesic circular orbit in equatorial lane
		if (V == -1 && projradius)
			V = 1 / (a +
			         r * sin(theta) * sqrt(r * sin(theta)));    // project the radius parallel to the equatorial plane
		else if (V == -1)
			V = 1 / (a + r * sqrt(r));

		// if(reverse) V = -1;

		// timelike basis vector
		et[0] = (1 / sqrt(e2nu)) / sqrt(1 - (V - omega) * (V - omega) * e2psi / e2nu);
		et[3] = (1 / sqrt(e2nu)) * V / sqrt(1 - (V - omega) * (V - omega) * e2psi / e2nu);
	}
	else if(motion == 1)
	{
		// if v < 0, use the speed relative to the local speed of light
		if(V < 0) V = abs(V) * (r*r - 2*r + spin+spin) / (r*r + spin*spin);

		et[0] = 1. / sqrt(g[0*4+0] + g[1*4+1]*V*V);
		et[1] = V * et[0];
	}

	// photon momentum
    momentum_from_consts<T>(p[0], p[1], p[2], p[3], k, h, Q, rdot_sign, thetadot_sign, r, theta, phi, spin);

	// if we're propagating backwards, reverse the direction of the photon momentum
	if(reverse)
	{
	    p[1] *= -1; p[2] *= -1; p[3] *= -1;
	}

	// evaluate dot product to get energy
	T recv = 0;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			recv += g[i*4 + j] * et[i] * p[j];

	return (reverse) ? recv / emit : emit / recv;
}


template <typename T>
inline T Raytracer<T>::ray_redshift( const T et[4], bool reverse, T r, T theta, T phi, T k, T h, T Q, int rdot_sign, int thetadot_sign, T emit )
{
	// calculate the redshift of a single ray given the observer 4-velocity et[4]

	T p[4];

	// metric coefficients
	const T rhosq = r*r + (spin*cos(theta))*(spin*cos(theta));
	const T delta = r*r - 2*r + spin*spin;
	const T sigmasq = (r*r + spin*spin)*(r*r + spin*spin) - spin*spin*delta*sin(theta)*sin(theta);

	const T e2nu = rhosq * delta / sigmasq;
	const T e2psi = sigmasq * sin(theta)*sin(theta) / rhosq;
	const T omega = 2*spin*r / sigmasq;

	T g[16];
	for(int i=0; i<16; i++)
		g[i] = 0;

	// g[i][j] -> g[i*4 + j]
	g[0*4 + 0] = e2nu - omega*omega*e2psi;
	g[0*4 + 3] = omega*e2psi;
	g[3*4 + 0] = g[0*4 + 3];
	g[1*4 + 1] = -rhosq/delta;
	g[2*4 + 2] = -rhosq;
	g[3*4 + 3] = -e2psi;

	// photon momentum
    momentum_from_consts<T>(p[0], p[1], p[2], p[3], k, h, Q, rdot_sign, thetadot_sign, r, theta, phi, spin);

	// if we're propagating backwards, reverse the direction of the photon momentum
	if(reverse)
	{
	    p[1] *= -1; p[2] *= -1; p[3] *= -1;
	}

	// evaluate dot product to get energy
	T recv = 0;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			recv += g[i*4 + j] * et[i] * p[j];

	return (reverse) ? recv / emit : emit / recv;
}


template <typename T>
void Raytracer<T>::range_phi(T min, T max )
{
	//
	// Puts the azimuthal angle co-ordinate (phi) in the required range
	//
	// Arguments:
	//	min		T		Lower bound of range (default value = -1 * M_PI)
	//	max		T		Upper bound of range (default value = M_PI)
	//
	cout << "Putting phi co-ordinate into range [" << min << "," << max << "]" << endl;
	for( int ray=0; ray<nRays; ray++ )
	{
		// check phi isn't something horrible so we don't enter an infinite loop
		if( abs(rays[ray].phi) > 1000 || rays[ray].phi != rays[ray].phi || !(rays[ray].steps>0) ) continue;

		while( rays[ray].phi >= max ) rays[ray].phi -= 2*M_PI;
		while( rays[ray].phi < min ) rays[ray].phi += 2*M_PI;
	}
}


template <typename T>
inline void Raytracer<T>::calculate_constants(int ray, T alpha, T beta, T V, T E)
{
	//
	// Compute the constants of motion for a ray emitted at polar angles alpha and beta in the frame
	// of a source at (rays[ray].t,rays[ray].r,rays[ray].theta,rays[ray].phi) orbiting azimuthally at angular velocity V
	//
	const T rhosq = rays[ray].r*rays[ray].r + (spin*cos(rays[ray].theta))*(spin*cos(rays[ray].theta));
	const T delta = rays[ray].r*rays[ray].r - 2*rays[ray].r + spin*spin;
	const T sigmasq = (rays[ray].r*rays[ray].r + spin*spin)*(rays[ray].r*rays[ray].r + spin*spin) - spin*spin*delta*sin(rays[ray].theta)*sin(rays[ray].theta);

	// metric coefficients
	const T e2nu = rhosq * delta / sigmasq;
	const T e2psi = sigmasq * sin(rays[ray].theta)*sin(rays[ray].theta) / rhosq;
	const T omega = 2*spin*rays[ray].r / sigmasq;

	// tetrad basis vector components
	const T et0 = (1/sqrt(e2nu))/sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu);
	const T et3 = (1/sqrt(e2nu))*V / sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu);
	//
	const T e10 = (V - omega)*sqrt(e2psi/e2nu) / sqrt(e2nu - (V-omega)*(V-omega)*e2psi);
	const T e13 = (1/sqrt(e2nu*e2psi))*(e2nu + V*omega*e2psi - omega*omega*e2psi) / sqrt(e2nu - (V-omega)*(V-omega)*e2psi);
	//
	const T e22 = -1/sqrt(rhosq);
	//
	const T e31 = sqrt(delta/rhosq);

	// photon 4-momentum in source frame
	const T rdotprime[] = { E, E*sin(alpha)*cos(beta), E*sin(alpha)*sin(beta), E*cos(alpha) };

	const T tdot = rdotprime[0]*et0 + rdotprime[1]*e10;
	const T phidot = rdotprime[0]*et3 + rdotprime[1]*e13;
	 const T rdot = rdotprime[3]*e31;
	const T thetadot = rdotprime[2]*e22;

	// find the corresponding values of k, h and Q using the geodesic equations
	rays[ray].k = (1 - 2*rays[ray].r/rhosq)*tdot + (2*spin*rays[ray].r*sin(rays[ray].theta)*sin(rays[ray].theta)/rhosq)*phidot;

	rays[ray].h = phidot * ( (rays[ray].r*rays[ray].r + spin*spin)*(rays[ray].r*rays[ray].r + spin*spin*cos(rays[ray].theta)*cos(rays[ray].theta) - 2*rays[ray].r)*sin(rays[ray].theta)*sin(rays[ray].theta) + 2*spin*spin*rays[ray].r*sin(rays[ray].theta)*sin(rays[ray].theta)*sin(rays[ray].theta)*sin(rays[ray].theta) );
	rays[ray].h = rays[ray].h - 2*spin*rays[ray].r*rays[ray].k*sin(rays[ray].theta)*sin(rays[ray].theta);
	rays[ray].h = rays[ray].h / ( rays[ray].r*rays[ray].r + spin*spin*cos(rays[ray].theta)*cos(rays[ray].theta) - 2*rays[ray].r );

	rays[ray].Q = rhosq*rhosq*thetadot*thetadot - (spin*rays[ray].k*cos(rays[ray].theta) + rays[ray].h/tan(rays[ray].theta))*(spin*rays[ray].k*cos(rays[ray].theta) - rays[ray].h/tan(rays[ray].theta));

	rays[ray].rdot_sign = (rdot >= 0) ? 1 : -1;
	rays[ray].thetadot_sign = (thetadot > 0) ? 1 : -1;
	rays[ray].rdot_flips = 0;
	rays[ray].equatorial_crossings = 0;

	//if(abs(rdot) < (1e-2 * e31)) rays[ray].steps = -1;
//	if(abs(rays[ray].r*phidot/rdot) > 1e3) rays[ray].steps = -1;
}

template <typename T>
inline void Raytracer<T>::calculate_constants_from_p(int ray, T pt, T pr, T ptheta, T pphi)
{
	//
	// calculate constants of motion from the 4-momentum and location of a photon
	//
	const T a = spin;
	const T r = rays[ray].r;
	const T theta = rays[ray].theta;

	const T rhosq = r*r + (a*cos(theta))*(a*cos(theta));

	T k = (1 - 2*r/rhosq)*pr + (2*a*r*sin(theta)*sin(theta)/rhosq)*pphi;

	T h = pphi * ( (r*r + a*a)*(r*r + a*a*cos(theta)*cos(theta) - 2*r)*sin(theta)*sin(theta) + 2*a*a*r*sin(theta)*sin(theta)*sin(theta)*sin(theta) );
	h = h - 2*a*r*k*sin(theta)*sin(theta);
	h = h / ( r*r + a*a*cos(theta)*cos(theta) - 2*r );

	T Q = rhosq*rhosq*ptheta*ptheta - (a*k*cos(theta) + h/tan(theta))*(a*k*cos(theta) - h/tan(theta));
	
	rays[ray].k = k;
	rays[ray].h = h;
	rays[ray].Q = Q;
}


template<typename T>
void Raytracer<T>::calculate_momentum( )
{
	//
	// Calculate photon momentum from constants of motion at a location
	//
	T a = spin;

	for(int ray = 0; ray < nRays; ray++)
	{
		const T t = rays[ray].t;
		const T r = rays[ray].r;
		const T theta = rays[ray].theta;
		const T phi = rays[ray].phi;
		const int rdot_sign = rays[ray].rdot_sign;
		const int thetadot_sign = rays[ray].thetadot_sign;

		const T k = rays[ray].k;
		const T h = rays[ray].h;
		const T Q = rays[ray].Q;

		const T sin_theta = sin(theta);
		const T cos_theta = cos(theta);
		const T sin2theta = sin_theta * sin_theta;
		const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
		const T delta = r*r - 2*r + a*a;
		const T rhosq_delta = rhosq * delta;

		// tdot
		rays[ray].pt = (rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h;
		rays[ray].pt /= rhosq_delta;

		// phidot
		rays[ray].pphi = 2*a*r*sin2theta*k + (rhosq - 2*r)*h;
		rays[ray].pphi /= sin2theta * rhosq_delta;

		// thetadot
		T thetadotsq = Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta);
		thetadotsq = thetadotsq / (rhosq*rhosq);

		// take the square roots and get the right signs
		rays[ray].ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

		// rdot
		T rdotsq = k*rays[ray].pt - h*rays[ray].pphi - rhosq*rays[ray].ptheta*rays[ray].ptheta;
		rdotsq = rdotsq * delta/rhosq;

		rays[ray].pr = sqrt(abs(rdotsq)) * rdot_sign;
	}
}

template <typename T>
inline int Raytracer<T>::propagate_rk4(int ray, const T rlim, const T thetalim, const int steplim, TextOutput* outfile, int write_step, T write_rmax, T write_rmin, bool write_cartesian)
{
	//
	// Propagate the photon along its geodesic using a classical 4th-order Runge-Kutta (RK4) integrator.
	//
	// At each step the photon momenta are evaluated at four trial positions (k1-k4) using the
	// conserved constants of motion (k, h, Q), then combined with the standard RK4 weights.
	// Sign tracking, step-size control, and termination conditions are identical to propagate().
	//
	int steps = 0;

	bool r_was_positive     = false;  // only allow r sign flip after rdotsq has been positive
	bool theta_was_positive = true;   // allow flip on first step if initial direction is wrong

	T rdotsq, thetadotsq;

	T step;

	T x, y, z;

	// copy variables locally
	T a = spin;
	T t = rays[ray].t;
	T r = rays[ray].r;
	T theta = rays[ray].theta;
	T phi = rays[ray].phi;
	T pt = rays[ray].pt;
	T pr = rays[ray].pr;
	T ptheta = rays[ray].ptheta;
	T pphi = rays[ray].pphi;
	int rdot_sign = rays[ray].rdot_sign;
	int thetadot_sign = rays[ray].thetadot_sign;
	int rdot_flips = rays[ray].rdot_flips;
	int equatorial_crossings = rays[ray].equatorial_crossings;
	T theta_eq_prev = theta;

	const T k = rays[ray].k;
	const T h = rays[ray].h;
	const T Q = rays[ray].Q;

	bool write_started = false;


	while( r < rlim  && ( (thetalim > 0 && theta < thetalim) || (thetalim < 0 && theta > abs(thetalim)) || thetalim == 0 )  &&  steps < steplim )
	{
		++steps;

		// === k1: evaluate momenta at current position (with sign-flip detection) ===

		const T sin_theta = sin(theta);
		const T cos_theta = cos(theta);
		const T sin2theta = sin_theta * sin_theta;
		const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
		const T delta = r*r - 2*r + a*a;
		const T rhosq_delta = rhosq * delta;

		// tdot (k1)
		pt = (rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h;
		pt /= rhosq_delta;

		// phidot (k1)
		pphi = 2*a*r*sin2theta*k + (rhosq - 2*r)*h;
		pphi /= sin2theta * rhosq_delta;

		// thetadot (k1) with sign-flip detection
		thetadotsq = Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta);
		thetadotsq = thetadotsq / (rhosq*rhosq);

		if (thetadotsq < 0 && theta_was_positive)
		{
			thetadot_sign *= -1;
			theta_was_positive = false;
			continue;
		}
		if (thetadotsq >= 0) theta_was_positive = true;

		ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

		// rdot (k1) with sign-flip detection
		rdotsq = k*pt - h*pphi - rhosq*ptheta*ptheta;
		rdotsq = rdotsq * delta/rhosq;

		if (rdotsq <= 0 && r_was_positive)
		{
			rdot_sign *= -1;
			r_was_positive = false;
			rdot_flips++;
		}
		else if (rdotsq > 0)
		{
			r_was_positive = true;
		}

		pr = sqrt(abs(rdotsq)) * rdot_sign;

		// store k1 momenta
		const T pt1 = pt, pr1 = pr, ptheta1 = ptheta, pphi1 = pphi;

		// === Step size (based on k1 momenta, identical to propagate()) ===
        step = abs((r - (T) horizon) / pr1) / precision;
        if(step > abs(theta / ptheta1) / precision)
        {
            step = abs(theta / ptheta1) / theta_precision;
        }
        if(max_tstep > 0 && r < maxtstep_rlim && step > abs(max_tstep / pt1))
        {
            step = abs(max_tstep / pt1);
        }
        if(max_phistep > 0 && step > abs(max_phistep / pphi1))
        {
            step = abs(max_phistep / pphi1);
        }
        if(step < MIN_STEP) step = MIN_STEP;

        if(rlim > 0 && r + pr1 * step > rlim) step = abs((rlim - r) / pr1);
        if(thetalim > 0 && theta + ptheta1 * step > thetalim) step = abs((thetalim - theta) / ptheta1);

		// ergosphere check (based on k1 tdot)
		if(pt1 <= 0)
		{
			rays[ray].status |= RAY_STATUS_ERGO;
            //rays[ray].steps = -1;
			//break;
		}

		// check that the dot product of the photon 4-momentum with the local timelike Killing vector is positive
		if((1 - 2*r/rhosq) * pt1 + (2*a*r*sin2theta/rhosq) * pphi1 < 0)
		{
			rays[ray].status |= RAY_STATUS_NEG_ENERGY;
			//rays[ray].steps = -1;
			//break;
		}

		// === k2: evaluate momenta at (r + step/2*pr1, theta + step/2*ptheta1) ===
		T pt2, pr2, ptheta2, pphi2;
		momentum_from_consts<T>(pt2, pr2, ptheta2, pphi2, k, h, Q,
		                        rdot_sign, thetadot_sign,
		                        r + (step/2)*pr1, theta + (step/2)*ptheta1, phi, a);

		// === k3: evaluate momenta at (r + step/2*pr2, theta + step/2*ptheta2) ===
		T pt3, pr3, ptheta3, pphi3;
		momentum_from_consts<T>(pt3, pr3, ptheta3, pphi3, k, h, Q,
		                        rdot_sign, thetadot_sign,
		                        r + (step/2)*pr2, theta + (step/2)*ptheta2, phi, a);

		// === k4: evaluate momenta at (r + step*pr3, theta + step*ptheta3) ===
		T pt4, pr4, ptheta4, pphi4;
		momentum_from_consts<T>(pt4, pr4, ptheta4, pphi4, k, h, Q,
		                        rdot_sign, thetadot_sign,
		                        r + step*pr3, theta + step*ptheta3, phi, a);

		// === RK4 weighted position update ===
		t     += (step / 6) * (pt1     + 2*pt2     + 2*pt3     + pt4);
		r     += (step / 6) * (pr1     + 2*pr2     + 2*pr3     + pr4);
		{ T tep = theta; theta += (step / 6) * (ptheta1 + 2*ptheta2 + 2*ptheta3 + ptheta4);
		  if ((tep < M_PI_2 && theta >= M_PI_2) || (tep > M_PI_2 && theta <= M_PI_2)) ++equatorial_crossings; }
		phi   += (step / 6) * (pphi1   + 2*pphi2   + 2*pphi3   + pphi4);
		// Reflect at polar axes: clamp theta to [0, pi]
		if (theta < T(0))    { theta = -theta;                thetadot_sign = -thetadot_sign; phi += T(M_PI); }
		if (theta > T(M_PI)) { theta = T(2)*T(M_PI) - theta; thetadot_sign = -thetadot_sign; phi += T(M_PI); }

		if(r <= horizon)
		{
			rays[ray].status |= RAY_STATUS_HORIZON;
			break;
		}

		if(outfile != 0 && (steps % write_step) == 0 )
		{
			if((write_rmax < 0 || r < write_rmax) && (write_rmin < 0 || r > write_rmin) )
			{
				write_started = true;
				if(write_cartesian)
				{
                    cartesian<T>(x, y, z, r, theta, phi, a);
					(*outfile) << t << x << y << z << endl;
				}
				else
				{
					(*outfile) << t << r << theta << phi << endl;
				}
			}
			else if(write_started)
			{
				break;
			}
		}
	}

	if (steps >= steplim)
		rays[ray].status |= RAY_STATUS_STEPLIM;
	else if (r >= rlim)
		rays[ray].status |= RAY_STATUS_RLIM;
	else if ((thetalim > 0 && theta >= thetalim) || (thetalim < 0 && theta <= abs(thetalim)))
		rays[ray].status |= RAY_STATUS_DEST;

	rays[ray].t = t;
    rays[ray].r = r;
    rays[ray].theta = theta;
    rays[ray].phi = phi;
    rays[ray].pt = pt;
    rays[ray].pr = pr;
    rays[ray].ptheta = ptheta;
    rays[ray].pphi = pphi;
    rays[ray].rdot_sign = rdot_sign;
    rays[ray].thetadot_sign = thetadot_sign;
    rays[ray].rdot_flips = rdot_flips;
    rays[ray].equatorial_crossings = equatorial_crossings;

	if(steps > 0) rays[ray].steps += steps;
	if (rays[ray].status & RAY_STATUS_STEPLIM)
		rays[ray].steps = -rays[ray].steps;   // negative steps flags a stuck/failed ray

	return steps;
}

template <typename T>
void Raytracer<T>::run_raytrace(RayDestination<T>* dest, Integrator method, T r_max,
                                int show_progress, TextOutput* outfile,
                                int write_step, T write_rmax, T write_rmin, bool write_cartesian,
                                int steplim)
{
    //
    // Unified ray tracing entry point with user-supplied stopping criterion.
    // Accepts a RayDestination object whose reached(r, theta, phi) method is called after each step.
    // Supports Integrator::RK4, Integrator::RK45 and Integrator::Symplectic; Euler has no RayDestination propagate variant.
    //
    assert(method != Integrator::Euler && "Integrator::Euler does not support RayDestination stopping conditions");
    static const char* names[] = {
        "",                              // Euler — not supported here
        "Running raytracer (RK4)...",
        "Running raytracer (RK45/DOPRI5)...",
        "Running raytracer (symplectic)..."
    };
    const int steplim = (method == Integrator::RK45) ? RK45_STEPLIM
                      : (method == Integrator::Symplectic) ? SYMP_STEPLIM : STEPLIM;
    cout << names[static_cast<int>(method)] << endl;

    ProgressBar prog(nRays, "Ray", 0, (show_progress > 0));
    show_progress = abs(show_progress);

    if (outfile != nullptr) {
        for (int ray = 0; ray < nRays; ray++)
        {
            if (show_progress != 0 && (ray % show_progress) == 0) prog.show(ray + 1);
            if (rays[ray].steps < 0) continue;
            else if (rays[ray].steps >= effective_steplim) continue;

            switch (method) {
                case Integrator::RK4:  propagate_rk4 (ray, r_max, dest, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::RK45: propagate_rk45(ray, r_max, dest, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::Symplectic: propagate_symplectic(ray, r_max, dest, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian); break;
                default: break;
            }
            outfile->newline(2);
        }
    } else {
        int rays_done = 0;
        #pragma omp parallel for schedule(dynamic)
        for (int ray = 0; ray < nRays; ray++)
        {
            if (show_progress != 0) {
                int done;
                #pragma omp atomic capture
                done = ++rays_done;
                if (done % show_progress == 0) {
                    #pragma omp critical
                    prog.show(done);
                }
            }
            if (rays[ray].steps < 0) continue;
            else if (rays[ray].steps >= effective_steplim) continue;

            switch (method) {
                case Integrator::RK4:  propagate_rk4 (ray, r_max, dest, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::RK45: propagate_rk45(ray, r_max, dest, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
                case Integrator::Symplectic: propagate_symplectic(ray, r_max, dest, steplim, nullptr, write_step, write_rmax, write_rmin, write_cartesian); break;
                default: break;
            }
        }
    }
    prog.done();
}

template <typename T>
inline int Raytracer<T>::propagate_rk4(int ray, const T rlim, RayDestination<T>* dest, const int steplim,
                                        TextOutput* outfile, int write_step,
                                        T write_rmax, T write_rmin, bool write_cartesian)
{
	//
	// Propagate the photon along its geodesic using a classical 4th-order Runge-Kutta (RK4) integrator.
	// Identical to propagate_rk4(ray, rlim, thetalim, ...) except the polar-angle stopping condition
	// is replaced by a call to dest->reached(r, theta, phi) after each position update.
	//
	int steps = 0;

	bool r_was_positive     = false;  // only allow r sign flip after rdotsq has been positive
	bool theta_was_positive = true;   // allow flip on first step if initial direction is wrong

	T rdotsq, thetadotsq;

	T step;

	T x, y, z;

	// copy variables locally
	T a = spin;
	T t = rays[ray].t;
	T r = rays[ray].r;
	T theta = rays[ray].theta;
	T phi = rays[ray].phi;
	T pt = rays[ray].pt;
	T pr = rays[ray].pr;
	T ptheta = rays[ray].ptheta;
	T pphi = rays[ray].pphi;
	int rdot_sign = rays[ray].rdot_sign;
	int thetadot_sign = rays[ray].thetadot_sign;
	int rdot_flips = rays[ray].rdot_flips;
	int equatorial_crossings = rays[ray].equatorial_crossings;
	T theta_eq_prev = theta;

	const T k = rays[ray].k;
	const T h = rays[ray].h;
	const T Q = rays[ray].Q;

	bool write_started = false;


	while( r < rlim && steps < steplim )
	{
		++steps;

		// === k1: evaluate momenta at current position (with sign-flip detection) ===

		const T sin_theta = sin(theta);
		const T cos_theta = cos(theta);
		const T sin2theta = sin_theta * sin_theta;
		const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
		const T delta = r*r - 2*r + a*a;
		const T rhosq_delta = rhosq * delta;

		// tdot (k1)
		pt = (rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h;
		pt /= rhosq_delta;

		// phidot (k1)
		pphi = 2*a*r*sin2theta*k + (rhosq - 2*r)*h;
		pphi /= sin2theta * rhosq_delta;

		// thetadot (k1) with sign-flip detection
		thetadotsq = Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta);
		thetadotsq = thetadotsq / (rhosq*rhosq);

		if (thetadotsq < 0 && theta_was_positive)
		{
			thetadot_sign *= -1;
			theta_was_positive = false;
			continue;
		}
		if (thetadotsq >= 0) theta_was_positive = true;

		ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

		// rdot (k1) with sign-flip detection
		rdotsq = k*pt - h*pphi - rhosq*ptheta*ptheta;
		rdotsq = rdotsq * delta/rhosq;

		if (rdotsq <= 0 && r_was_positive)
		{
			rdot_sign *= -1;
			r_was_positive = false;
			rdot_flips++;
		}
		else if (rdotsq > 0)
		{
			r_was_positive = true;
		}

		pr = sqrt(abs(rdotsq)) * rdot_sign;

		// store k1 momenta
		const T pt1 = pt, pr1 = pr, ptheta1 = ptheta, pphi1 = pphi;

		// === Step size (based on k1 momenta) ===
        step = abs((r - (T) horizon) / pr1) / precision;
        if(step > abs(theta / ptheta1) / precision)
        {
            step = abs(theta / ptheta1) / theta_precision;
        }
        if(max_tstep > 0 && r < maxtstep_rlim && step > abs(max_tstep / pt1))
        {
            step = abs(max_tstep / pt1);
        }
        if(max_phistep > 0 && step > abs(max_phistep / pphi1))
        {
            step = abs(max_phistep / pphi1);
        }
        if(step < MIN_STEP) step = MIN_STEP;

        if(rlim > 0 && r + pr1 * step > rlim) step = abs((rlim - r) / pr1);

		// ergosphere check (based on k1 tdot)
		if(pt1 <= 0)
		{
			rays[ray].status |= RAY_STATUS_ERGO;
            // rays[ray].steps = -1;
			// break;
		}

		// check that the dot product of the photon 4-momentum with the local timelike Killing vector is positive
		if((1 - 2*r/rhosq) * pt1 + (2*a*r*sin2theta/rhosq) * pphi1 < 0)
		{
			rays[ray].status |= RAY_STATUS_NEG_ENERGY;
			// rays[ray].steps = -1;
			// break;
		}

		// === k2: evaluate momenta at (r + step/2*pr1, theta + step/2*ptheta1) ===
		T pt2, pr2, ptheta2, pphi2;
		momentum_from_consts<T>(pt2, pr2, ptheta2, pphi2, k, h, Q,
		                        rdot_sign, thetadot_sign,
		                        r + (step/2)*pr1, theta + (step/2)*ptheta1, phi, a);

		// === k3: evaluate momenta at (r + step/2*pr2, theta + step/2*ptheta2) ===
		T pt3, pr3, ptheta3, pphi3;
		momentum_from_consts<T>(pt3, pr3, ptheta3, pphi3, k, h, Q,
		                        rdot_sign, thetadot_sign,
		                        r + (step/2)*pr2, theta + (step/2)*ptheta2, phi, a);

		// === k4: evaluate momenta at (r + step*pr3, theta + step*ptheta3) ===
		T pt4, pr4, ptheta4, pphi4;
		momentum_from_consts<T>(pt4, pr4, ptheta4, pphi4, k, h, Q,
		                        rdot_sign, thetadot_sign,
		                        r + step*pr3, theta + step*ptheta3, phi, a);

		// === RK4 weighted position update ===
		const T theta_prev = theta;
		t     += (step / 6) * (pt1     + 2*pt2     + 2*pt3     + pt4);
		r     += (step / 6) * (pr1     + 2*pr2     + 2*pr3     + pr4);
		{ T tep = theta; theta += (step / 6) * (ptheta1 + 2*ptheta2 + 2*ptheta3 + ptheta4);
		  if ((tep < M_PI_2 && theta >= M_PI_2) || (tep > M_PI_2 && theta <= M_PI_2)) ++equatorial_crossings; }
		phi   += (step / 6) * (pphi1   + 2*pphi2   + 2*pphi3   + pphi4);
		// Reflect at polar axes: clamp theta to [0, pi]
		if (theta < T(0))    { theta = -theta;                thetadot_sign = -thetadot_sign; phi += T(M_PI); }
		if (theta > T(M_PI)) { theta = T(2)*T(M_PI) - theta; thetadot_sign = -thetadot_sign; phi += T(M_PI); }

		if(r <= horizon)
		{
			rays[ray].status |= RAY_STATUS_HORIZON;
			break;
		}
		if(dest->reached(r, theta, phi, theta_prev))
		{
			rays[ray].status |= RAY_STATUS_DEST;
			break;
		}

		if(outfile != 0 && (steps % write_step) == 0 )
		{
			if((write_rmax < 0 || r < write_rmax) && (write_rmin < 0 || r > write_rmin) )
			{
				write_started = true;
				if(write_cartesian)
				{
                    cartesian<T>(x, y, z, r, theta, phi, a);
					(*outfile) << t << x << y << z << endl;
				}
				else
				{
					(*outfile) << t << r << theta << phi << endl;
				}
			}
			else if(write_started)
			{
				break;
			}
		}
	}

	if (steps >= steplim)
			rays[ray].status |= RAY_STATUS_STEPLIM;
	else if (r >= rlim)
			rays[ray].status |= RAY_STATUS_RLIM;

	rays[ray].t = t;
    rays[ray].r = r;
    rays[ray].theta = theta;
    rays[ray].phi = phi;
    rays[ray].pt = pt;
    rays[ray].pr = pr;
    rays[ray].ptheta = ptheta;
    rays[ray].pphi = pphi;
    rays[ray].rdot_sign = rdot_sign;
    rays[ray].thetadot_sign = thetadot_sign;
    rays[ray].rdot_flips = rdot_flips;
    rays[ray].equatorial_crossings = equatorial_crossings;

	if(steps > 0) rays[ray].steps += steps;
	if (rays[ray].status & RAY_STATUS_STEPLIM)
		rays[ray].steps = -rays[ray].steps;   // negative steps flags a stuck/failed ray

	return steps;
}

// =============================================================================
// RK45 / Dormand-Prince (DOPRI5) adaptive integrator
// =============================================================================

template <typename T>
inline int Raytracer<T>::propagate_rk45(int ray, const T rlim, const T thetalim, const int steplim,
                                         TextOutput* outfile, int write_step,
                                         T write_rmax, T write_rmin, bool write_cartesian)
{
    //
    // Propagate the photon along its geodesic using the adaptive Dormand-Prince RK45 integrator.
    //
    // Algorithm: DOPRI5 (Dormand & Prince, 1980, J. Comput. Appl. Math. 6, 19-26).
    // An embedded 4th/5th-order Runge-Kutta pair shares a single 7-stage evaluation.
    // The 5th-order solution is propagated (local extrapolation) and its difference from
    // the embedded 4th-order estimate gives a per-step local error.  The step size is
    // adjusted each step to keep this normalised error below 1e-8 (mixed abs/rel norm
    // over r and theta).
    //
    // Step-size controller (Hairer & Wanner, "Solving ODEs I", §II.4):
    //   step_new = step * min(5, max(0.1, 0.9 * (1/err)^(1/5)))
    //
    // The FSAL (First Same As Last) optimisation — reusing k7 as k1 of the next step —
    // is not implemented here; k1 is recomputed from the constants of motion at the start
    // of each step.  This costs one extra derivative evaluation per step but keeps the
    // sign-flip logic simple and consistent with propagate() and propagate_rk4().
    //

    int steps = 0;

    bool r_was_positive     = false;  // only allow r sign flip after rdotsq has been positive
    bool theta_was_positive = true;   // allow flip on first step if initial direction is wrong

    T x, y, z;

    T a      = spin;
    T t      = rays[ray].t;
    T r      = rays[ray].r;
    T theta  = rays[ray].theta;
    T phi    = rays[ray].phi;
    T pt     = rays[ray].pt;
    T pr     = rays[ray].pr;
    T ptheta = rays[ray].ptheta;
    T pphi   = rays[ray].pphi;
    int rdot_sign     = rays[ray].rdot_sign;
    int thetadot_sign = rays[ray].thetadot_sign;
    int rdot_flips    = rays[ray].rdot_flips;
    int equatorial_crossings = rays[ray].equatorial_crossings;
    T theta_eq_prev = theta;

    const T k = rays[ray].k;
    const T h = rays[ray].h;   // z-angular momentum constant of motion
    const T Q = rays[ray].Q;

    bool write_started = false;

    // -------------------------------------------------------------------------
    // DOPRI5 Butcher tableau (Dormand & Prince 1980, Table 2, p.21)
    // Nodes: c = {0, 1/5, 3/10, 4/5, 8/9, 1, 1}
    // -------------------------------------------------------------------------
    static constexpr T a21 = T(1)/5;
    static constexpr T a31 = T(3)/40,         a32 = T(9)/40;
    static constexpr T a41 = T(44)/45,         a42 = T(-56)/15,        a43 = T(32)/9;
    static constexpr T a51 = T(19372)/6561,    a52 = T(-25360)/2187,   a53 = T(64448)/6561,  a54 = T(-212)/729;
    static constexpr T a61 = T(9017)/3168,     a62 = T(-355)/33,       a63 = T(46732)/5247,
                        a64 = T(49)/176,        a65 = T(-5103)/18656;

    // 5th-order propagation weights (b2 = 0, so k2 does not enter the solution):
    static constexpr T b1 = T(35)/384,    b3 = T(500)/1113,   b4 = T(125)/192,
                        b5 = T(-2187)/6784, b6 = T(11)/84;

    // Error coefficients e_i = b_i - b*_i (difference from embedded 4th-order weights).
    // These use the k7 FSAL stage evaluated at the 5th-order estimate:
    static constexpr T e1 = T(71)/57600,      e3 = T(-71)/16695,     e4 = T(71)/1920,
                        e5 = T(-17253)/339200, e6 = T(22)/525,        e7 = T(-1)/40;
    // -------------------------------------------------------------------------

    // Adaptive step-size control parameters (Hairer & Wanner §II.4)
    static constexpr T safety  = T(0.9);    // conservative scale on predicted step
    static constexpr T fac_max = T(5.0);    // cap on per-step growth
    static constexpr T fac_min = T(0.1);    // floor on per-step shrink
    const            T tol     = rk45_tol;  // mixed abs/rel error tolerance; tunable via set_rk45_tol()

    // Seed the running step size with the same heuristic as propagate()/propagate_rk4().
    // The adaptive controller will adjust this from the very first step.
    {
        const T sin_theta = sin(theta), cos_theta = cos(theta), sin2theta = sin_theta * sin_theta;
        const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
        const T delta  = r*r - 2*r + a*a;
        pt   = ((rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h) / (rhosq * delta);
        pphi = (2*a*r*sin2theta*k + (rhosq - 2*r)*h) / (sin2theta * rhosq * delta);
        T thetadotsq = (Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta)) / (rhosq*rhosq);
        ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;
        T rdotsq = (k*pt - h*pphi - rhosq*ptheta*ptheta) * delta / rhosq;
        pr = sqrt(abs(rdotsq)) * rdot_sign;
    }
    T step = abs((r - (T)horizon) / pr) / precision;
    if (abs(ptheta) > 0 && step > abs(theta / ptheta) / theta_precision)
        step = abs(theta / ptheta) / theta_precision;
    if (max_tstep > 0 && r < maxtstep_rlim && step > abs(max_tstep / pt))
        step = abs(max_tstep / pt);
    if (max_phistep > 0 && step > abs(max_phistep / pphi))
        step = abs(max_phistep / pphi);
    if (step < MIN_STEP) step = MIN_STEP;

    // --- Main integration loop ---
    while (r < rlim
           && ((thetalim > 0 && theta < thetalim) || (thetalim < 0 && theta > abs(thetalim)) || thetalim == 0)
           && steps < steplim)
    {
        ++steps;

        // === k1: momenta at current position, with sign-flip detection ===
        {
            const T sin_theta = sin(theta), cos_theta = cos(theta), sin2theta = sin_theta * sin_theta;
            const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
            const T delta  = r*r - 2*r + a*a;

            pt   = ((rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h) / (rhosq * delta);
            pphi = (2*a*r*sin2theta*k + (rhosq - 2*r)*h) / (sin2theta * rhosq * delta);

            T thetadotsq = (Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta)) / (rhosq*rhosq);
            if (thetadotsq < 0 && theta_was_positive)
            {
                thetadot_sign *= -1;
                theta_was_positive = false;
                continue;
            }
            if (thetadotsq >= 0) theta_was_positive = true;
            ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

            T rdotsq = (k*pt - h*pphi - rhosq*ptheta*ptheta) * delta / rhosq;
            if (rdotsq <= 0 && r_was_positive)
            {
                rdot_sign *= -1;
                r_was_positive = false;
                rdot_flips++;
            }
            else if (rdotsq > 0)
            {
                r_was_positive = true;
            }
            pr = sqrt(abs(rdotsq)) * rdot_sign;
        }

        const T pt1 = pt, pr1 = pr, ptheta1 = ptheta, pphi1 = pphi;

        if (pt1 <= 0)
            rays[ray].status |= RAY_STATUS_ERGO;
        {
            const T sin_theta = sin(theta), sin2theta = sin_theta * sin_theta;
            const T rhosq = r*r + (a*cos(theta))*(a*cos(theta));
            if ((1 - 2*r/rhosq)*pt1 + (2*a*r*sin2theta/rhosq)*pphi1 < 0)
                rays[ray].status |= RAY_STATUS_NEG_ENERGY;
        }

        // --- Cap the adaptive step to prevent intermediate DOPRI5 stages from crossing the
        //     event horizon.  The Butcher tableau has large negative coefficients (a42=-56/15,
        //     a52=-25360/2187) that can push intermediate trial positions inside the horizon
        //     when the step grows too large.  The error estimator misses this because both
        //     4th- and 5th-order solutions share the corrupted intermediate stages.
        //     Only the horizon cap is applied here.  The theta cap (used in the RK4 initial-
        //     step seeding) is intentionally omitted: applying it on every outer iteration
        //     would freeze the adaptive step at a tiny value for source geometries near the
        //     polar axis (theta ~ 0), causing rays to hit the step-count limit.
        {
            T step_max = abs((r - (T)horizon) / pr1) / precision;
            if (max_phistep > 0)
            {
                T step_phi = abs(max_phistep / pphi1);
                if (step_phi < step_max) step_max = step_phi;
            }
            if (max_tstep > 0 && r < maxtstep_rlim)
            {
                T step_t = abs(max_tstep / pt1);
                if (step_t < step_max) step_max = step_t;
            }
            if (step > step_max) step = step_max;
        }

        // --- Adaptive sub-loop: retry with smaller step until normalised error <= 1 ---
        bool step_accepted = false;
        while (!step_accepted)
        {
            // Clamp the trial step to avoid overshooting hard boundaries.
            // A clamped step is accepted normally but does not update the running step size.
            T h_try  = step;
            bool clamped = false;
            // if (rlim > 0 && r + pr1 * h_try > rlim)
            // {
            //     h_try  = abs((rlim - r) / pr1);
            //     clamped = true;
            // }
            if (thetalim > 0 && theta + ptheta1 * h_try > thetalim)
            {
                T h_th = abs((thetalim - theta) / ptheta1);
                if (h_th < h_try) { h_try = h_th; clamped = true; }
            }

            // === k2 ===
            T pt2, pr2, ptheta2, pphi2;
            momentum_from_consts<T>(pt2, pr2, ptheta2, pphi2, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*a21*pr1,
                                    theta + h_try*a21*ptheta1, phi, a);

            // === k3 ===
            T pt3, pr3, ptheta3, pphi3;
            momentum_from_consts<T>(pt3, pr3, ptheta3, pphi3, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a31*pr1     + a32*pr2),
                                    theta + h_try*(a31*ptheta1 + a32*ptheta2), phi, a);

            // === k4 ===
            T pt4, pr4, ptheta4, pphi4;
            momentum_from_consts<T>(pt4, pr4, ptheta4, pphi4, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a41*pr1     + a42*pr2     + a43*pr3),
                                    theta + h_try*(a41*ptheta1 + a42*ptheta2 + a43*ptheta3), phi, a);

            // === k5 ===
            T pt5, pr5, ptheta5, pphi5;
            momentum_from_consts<T>(pt5, pr5, ptheta5, pphi5, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a51*pr1     + a52*pr2     + a53*pr3     + a54*pr4),
                                    theta + h_try*(a51*ptheta1 + a52*ptheta2 + a53*ptheta3 + a54*ptheta4),
                                    phi, a);

            // === k6 ===
            T pt6, pr6, ptheta6, pphi6;
            momentum_from_consts<T>(pt6, pr6, ptheta6, pphi6, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a61*pr1     + a62*pr2     + a63*pr3     + a64*pr4     + a65*pr5),
                                    theta + h_try*(a61*ptheta1 + a62*ptheta2 + a63*ptheta3 + a64*ptheta4 + a65*ptheta5),
                                    phi, a);

            // === 5th-order solution (local extrapolation; b2=0 so k2 is absent) ===
            T r_new     = r     + h_try*(b1*pr1     + b3*pr3     + b4*pr4     + b5*pr5     + b6*pr6);
            T theta_new = theta + h_try*(b1*ptheta1  + b3*ptheta3  + b4*ptheta4  + b5*ptheta5  + b6*ptheta6);
            T t_new     = t     + h_try*(b1*pt1     + b3*pt3     + b4*pt4     + b5*pt5     + b6*pt6);
            T phi_new   = phi   + h_try*(b1*pphi1   + b3*pphi3   + b4*pphi4   + b5*pphi5   + b6*pphi6);
            // Reflect at polar axes: clamp theta_new to [0, pi]
            if (theta_new < T(0))    { theta_new = -theta_new;                thetadot_sign = -thetadot_sign; phi_new += T(M_PI); }
            if (theta_new > T(M_PI)) { theta_new = T(2)*T(M_PI) - theta_new; thetadot_sign = -thetadot_sign; phi_new += T(M_PI); }

            // === k7: FSAL evaluation at the 5th-order estimate (needed for error estimate) ===
            T pt7, pr7, ptheta7, pphi7;
            momentum_from_consts<T>(pt7, pr7, ptheta7, pphi7, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r_new, theta_new, phi_new, a);

            // === Error estimate: e_i = b_i - b*_i ===
            T err_r     = h_try*(e1*pr1     + e3*pr3     + e4*pr4     + e5*pr5     + e6*pr6     + e7*pr7);
            T err_theta = h_try*(e1*ptheta1  + e3*ptheta3  + e4*ptheta4  + e5*ptheta5  + e6*ptheta6  + e7*ptheta7);

            // Mixed absolute/relative norm; +1 in sc prevents blow-up near theta=0
            T sc_r     = tol * (T(1) + max(abs(r),     abs(r_new)));
            T sc_theta = tol * (T(1) + max(abs(theta), abs(theta_new)));
            T err_norm = sqrt(T(0.5)*((err_r/sc_r)*(err_r/sc_r) + (err_theta/sc_theta)*(err_theta/sc_theta)));

            // === Step-size prediction ===
            T fac      = safety * pow(T(1) / max(err_norm, T(1e-10)), T(0.2));
            fac        = max(fac_min, min(fac_max, fac));
            T step_new = h_try * fac;

            if (err_norm <= T(1))
            {
                // Step accepted: commit new state
                t = t_new; r = r_new; theta = theta_new; phi = phi_new;
                pt = pt7; pr = pr7; ptheta = ptheta7; pphi = pphi7;
                if (!clamped) step = max(step_new, (T)MIN_STEP);
                step_accepted = true;
            }
            else
            {
                // Step rejected: shrink and retry
                step = max(step_new, (T)MIN_STEP);
                if (step <= (T)MIN_STEP)
                {
                    // Cannot shrink further; force-accept to avoid an infinite loop
                    t = t_new; r = r_new; theta = theta_new; phi = phi_new;
                    pt = pt7; pr = pr7; ptheta = ptheta7; pphi = pphi7;
                    step_accepted = true;
                }
            }
        }  // end adaptive sub-loop
        if ((theta_eq_prev < M_PI_2 && theta >= M_PI_2) || (theta_eq_prev > M_PI_2 && theta <= M_PI_2))
            ++equatorial_crossings;
        theta_eq_prev = theta;

        if (r <= horizon)
        {
            rays[ray].status |= RAY_STATUS_HORIZON;
            break;
        }

        if (outfile != nullptr && (steps % write_step) == 0)
        {
            if ((write_rmax < 0 || r < write_rmax) && (write_rmin < 0 || r > write_rmin))
            {
                write_started = true;
                if (write_cartesian)
                {
                    cartesian<T>(x, y, z, r, theta, phi, a);
                    (*outfile) << t << x << y << z << endl;
                }
                else
                {
                    (*outfile) << t << r << theta << phi << endl;
                }
            }
            else if (write_started)
            {
                break;
            }
        }
    }  // end main loop

    if (steps >= steplim)
        rays[ray].status |= RAY_STATUS_STEPLIM;
    else if (r >= rlim)
        rays[ray].status |= RAY_STATUS_RLIM;
    else if ((thetalim > 0 && theta >= thetalim) || (thetalim < 0 && theta <= abs(thetalim)))
        rays[ray].status |= RAY_STATUS_DEST;

    rays[ray].t           = t;
    rays[ray].r           = r;
    rays[ray].theta       = theta;
    rays[ray].phi         = phi;
    rays[ray].pt          = pt;
    rays[ray].pr          = pr;
    rays[ray].ptheta      = ptheta;
    rays[ray].pphi        = pphi;
    rays[ray].rdot_sign   = rdot_sign;
    rays[ray].thetadot_sign = thetadot_sign;
    rays[ray].rdot_flips   = rdot_flips;
    rays[ray].equatorial_crossings = equatorial_crossings;

    if (steps > 0) rays[ray].steps += steps;
    if (rays[ray].status & RAY_STATUS_STEPLIM)
        rays[ray].steps = -rays[ray].steps;   // negative steps flags a stuck/failed ray
    return steps;
}

template <typename T>
inline int Raytracer<T>::propagate_rk45(int ray, const T rlim, RayDestination<T>* dest, const int steplim,
                                         TextOutput* outfile, int write_step,
                                         T write_rmax, T write_rmin, bool write_cartesian)
{
    //
    // Identical to propagate_rk45(ray, rlim, thetalim, ...) except the polar-angle stopping
    // condition is replaced by a call to dest->reached(r, theta, phi) after each accepted step.
    //
    int steps = 0;

    bool r_was_positive     = false;  // only allow r sign flip after rdotsq has been positive
    bool theta_was_positive = true;   // allow flip on first step if initial direction is wrong

    T x, y, z;

    T a      = spin;
    T t      = rays[ray].t;
    T r      = rays[ray].r;
    T theta  = rays[ray].theta;
    T phi    = rays[ray].phi;
    T pt     = rays[ray].pt;
    T pr     = rays[ray].pr;
    T ptheta = rays[ray].ptheta;
    T pphi   = rays[ray].pphi;
    int rdot_sign     = rays[ray].rdot_sign;
    int thetadot_sign = rays[ray].thetadot_sign;
    int rdot_flips    = rays[ray].rdot_flips;
    int equatorial_crossings = rays[ray].equatorial_crossings;
    T theta_eq_prev = theta;

    const T k = rays[ray].k;
    const T h = rays[ray].h;
    const T Q = rays[ray].Q;

    bool write_started = false;

    // DOPRI5 tableau (same constants as propagate_rk45 theta_max overload)
    static constexpr T a21 = T(1)/5;
    static constexpr T a31 = T(3)/40,         a32 = T(9)/40;
    static constexpr T a41 = T(44)/45,         a42 = T(-56)/15,        a43 = T(32)/9;
    static constexpr T a51 = T(19372)/6561,    a52 = T(-25360)/2187,   a53 = T(64448)/6561,  a54 = T(-212)/729;
    static constexpr T a61 = T(9017)/3168,     a62 = T(-355)/33,       a63 = T(46732)/5247,
                        a64 = T(49)/176,        a65 = T(-5103)/18656;
    static constexpr T b1 = T(35)/384,    b3 = T(500)/1113,   b4 = T(125)/192,
                        b5 = T(-2187)/6784, b6 = T(11)/84;
    static constexpr T e1 = T(71)/57600,      e3 = T(-71)/16695,     e4 = T(71)/1920,
                        e5 = T(-17253)/339200, e6 = T(22)/525,        e7 = T(-1)/40;

    static constexpr T safety  = T(0.9);
    static constexpr T fac_max = T(5.0);
    static constexpr T fac_min = T(0.1);
    const            T tol     = rk45_tol;  // tunable via set_rk45_tol()

    {
        const T sin_theta = sin(theta), cos_theta = cos(theta), sin2theta = sin_theta * sin_theta;
        const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
        const T delta  = r*r - 2*r + a*a;
        pt   = ((rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h) / (rhosq * delta);
        pphi = (2*a*r*sin2theta*k + (rhosq - 2*r)*h) / (sin2theta * rhosq * delta);
        T thetadotsq = (Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta)) / (rhosq*rhosq);
        ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;
        T rdotsq = (k*pt - h*pphi - rhosq*ptheta*ptheta) * delta / rhosq;
        pr = sqrt(abs(rdotsq)) * rdot_sign;
    }
    T step = abs((r - (T)horizon) / pr) / precision;
    if (abs(ptheta) > 0 && step > abs(theta / ptheta) / theta_precision)
        step = abs(theta / ptheta) / theta_precision;
    if (max_tstep > 0 && r < maxtstep_rlim && step > abs(max_tstep / pt))
        step = abs(max_tstep / pt);
    if (max_phistep > 0 && step > abs(max_phistep / pphi))
        step = abs(max_phistep / pphi);
    if (step < MIN_STEP) step = MIN_STEP;

    while (r < rlim && steps < steplim)
    {
        ++steps;

        // === k1 with sign-flip detection ===
        {
            const T sin_theta = sin(theta), cos_theta = cos(theta), sin2theta = sin_theta * sin_theta;
            const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
            const T delta  = r*r - 2*r + a*a;

            pt   = ((rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h) / (rhosq * delta);
            pphi = (2*a*r*sin2theta*k + (rhosq - 2*r)*h) / (sin2theta * rhosq * delta);

            T thetadotsq = (Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta)) / (rhosq*rhosq);
            if (thetadotsq < 0 && theta_was_positive)
            {
                thetadot_sign *= -1;
                theta_was_positive = false;
                continue;
            }
            if (thetadotsq >= 0) theta_was_positive = true;
            ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

            T rdotsq = (k*pt - h*pphi - rhosq*ptheta*ptheta) * delta / rhosq;
            if (rdotsq <= 0 && r_was_positive)
            {
                rdot_sign *= -1;
                r_was_positive = false;
                rdot_flips++;
            }
            else if (rdotsq > 0)
            {
                r_was_positive = true;
            }
            pr = sqrt(abs(rdotsq)) * rdot_sign;
        }

        const T pt1 = pt, pr1 = pr, ptheta1 = ptheta, pphi1 = pphi;

        if (pt1 <= 0)
            rays[ray].status |= RAY_STATUS_ERGO;
        {
            const T sin_theta = sin(theta), sin2theta = sin_theta * sin_theta;
            const T rhosq = r*r + (a*cos(theta))*(a*cos(theta));
            if ((1 - 2*r/rhosq)*pt1 + (2*a*r*sin2theta/rhosq)*pphi1 < 0)
                rays[ray].status |= RAY_STATUS_NEG_ENERGY;
        }

        // --- Cap the adaptive step to prevent intermediate DOPRI5 stages from crossing the
        //     event horizon.  dest->step_limit() supplies an optional boundary-surface cap
        //     (analogous to the thetalim clamp in propagate_rk45); destinations that cannot
        //     define a meaningful limit return std::numeric_limits<T>::max() (no cap).
        {
            T step_max = abs((r - (T)horizon) / pr1) / precision;
            if (max_phistep > 0)
            {
                T step_phi = abs(max_phistep / pphi1);
                if (step_phi < step_max) step_max = step_phi;
            }
            if (max_tstep > 0 && r < maxtstep_rlim)
            {
                T step_t = abs(max_tstep / pt1);
                if (step_t < step_max) step_max = step_t;
            }
            if (step > step_max) step = step_max;
        }

        const T theta_prev = theta;
        bool step_accepted = false;
        while (!step_accepted)
        {
            T h_try  = step;
            bool clamped = false;
            if (rlim > 0 && r + pr1 * h_try > rlim)
            {
                h_try  = abs((rlim - r) / pr1);
                clamped = true;
            }
            {
                T h_dest = dest->step_limit(r, theta, phi, pr1, ptheta1, pphi1);
                if (h_dest < h_try) { h_try = h_dest; clamped = true; }
            }

            // === k2-k6 ===
            T pt2, pr2, ptheta2, pphi2;
            momentum_from_consts<T>(pt2, pr2, ptheta2, pphi2, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*a21*pr1,
                                    theta + h_try*a21*ptheta1, phi, a);

            T pt3, pr3, ptheta3, pphi3;
            momentum_from_consts<T>(pt3, pr3, ptheta3, pphi3, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a31*pr1     + a32*pr2),
                                    theta + h_try*(a31*ptheta1 + a32*ptheta2), phi, a);

            T pt4, pr4, ptheta4, pphi4;
            momentum_from_consts<T>(pt4, pr4, ptheta4, pphi4, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a41*pr1     + a42*pr2     + a43*pr3),
                                    theta + h_try*(a41*ptheta1 + a42*ptheta2 + a43*ptheta3), phi, a);

            T pt5, pr5, ptheta5, pphi5;
            momentum_from_consts<T>(pt5, pr5, ptheta5, pphi5, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a51*pr1     + a52*pr2     + a53*pr3     + a54*pr4),
                                    theta + h_try*(a51*ptheta1 + a52*ptheta2 + a53*ptheta3 + a54*ptheta4),
                                    phi, a);

            T pt6, pr6, ptheta6, pphi6;
            momentum_from_consts<T>(pt6, pr6, ptheta6, pphi6, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r + h_try*(a61*pr1     + a62*pr2     + a63*pr3     + a64*pr4     + a65*pr5),
                                    theta + h_try*(a61*ptheta1 + a62*ptheta2 + a63*ptheta3 + a64*ptheta4 + a65*ptheta5),
                                    phi, a);

            // === 5th-order solution ===
            T r_new     = r     + h_try*(b1*pr1     + b3*pr3     + b4*pr4     + b5*pr5     + b6*pr6);
            T theta_new = theta + h_try*(b1*ptheta1  + b3*ptheta3  + b4*ptheta4  + b5*ptheta5  + b6*ptheta6);
            T t_new     = t     + h_try*(b1*pt1     + b3*pt3     + b4*pt4     + b5*pt5     + b6*pt6);
            T phi_new   = phi   + h_try*(b1*pphi1   + b3*pphi3   + b4*pphi4   + b5*pphi5   + b6*pphi6);
            // Reflect at polar axes: clamp theta_new to [0, pi]
            if (theta_new < T(0))    { theta_new = -theta_new;                thetadot_sign = -thetadot_sign; phi_new += T(M_PI); }
            if (theta_new > T(M_PI)) { theta_new = T(2)*T(M_PI) - theta_new; thetadot_sign = -thetadot_sign; phi_new += T(M_PI); }

            // === k7 (FSAL) ===
            T pt7, pr7, ptheta7, pphi7;
            momentum_from_consts<T>(pt7, pr7, ptheta7, pphi7, k, h, Q,
                                    rdot_sign, thetadot_sign,
                                    r_new, theta_new, phi_new, a);

            // === Error estimate ===
            T err_r     = h_try*(e1*pr1     + e3*pr3     + e4*pr4     + e5*pr5     + e6*pr6     + e7*pr7);
            T err_theta = h_try*(e1*ptheta1  + e3*ptheta3  + e4*ptheta4  + e5*ptheta5  + e6*ptheta6  + e7*ptheta7);

            T sc_r     = tol * (T(1) + max(abs(r),     abs(r_new)));
            T sc_theta = tol * (T(1) + max(abs(theta), abs(theta_new)));
            T err_norm = sqrt(T(0.5)*((err_r/sc_r)*(err_r/sc_r) + (err_theta/sc_theta)*(err_theta/sc_theta)));

            T fac      = safety * pow(T(1) / max(err_norm, T(1e-10)), T(0.2));
            fac        = max(fac_min, min(fac_max, fac));
            T step_new = h_try * fac;

            if (err_norm <= T(1))
            {
                t = t_new; r = r_new; theta = theta_new; phi = phi_new;
                pt = pt7; pr = pr7; ptheta = ptheta7; pphi = pphi7;
                if (!clamped) step = max(step_new, (T)MIN_STEP);
                step_accepted = true;
            }
            else
            {
                step = max(step_new, (T)MIN_STEP);
                if (step <= (T)MIN_STEP)
                {
                    t = t_new; r = r_new; theta = theta_new; phi = phi_new;
                    pt = pt7; pr = pr7; ptheta = ptheta7; pphi = pphi7;
                    step_accepted = true;
                }
            }
        }
        if ((theta_eq_prev < M_PI_2 && theta >= M_PI_2) || (theta_eq_prev > M_PI_2 && theta <= M_PI_2))
            ++equatorial_crossings;
        theta_eq_prev = theta;

        if (r <= horizon)
        {
            rays[ray].status |= RAY_STATUS_HORIZON;
            break;
        }
        if (dest->reached(r, theta, phi, theta_prev))
        {
            rays[ray].status |= RAY_STATUS_DEST;
            break;
        }

        if (outfile != nullptr && (steps % write_step) == 0)
        {
            if ((write_rmax < 0 || r < write_rmax) && (write_rmin < 0 || r > write_rmin))
            {
                write_started = true;
                if (write_cartesian)
                {
                    cartesian<T>(x, y, z, r, theta, phi, a);
                    (*outfile) << t << x << y << z << endl;
                }
                else
                {
                    (*outfile) << t << r << theta << phi << endl;
                }
            }
            else if (write_started)
            {
                break;
            }
        }
    }

    if (steps >= steplim)
        rays[ray].status |= RAY_STATUS_STEPLIM;
    else if (r >= rlim)
        rays[ray].status |= RAY_STATUS_RLIM;

    rays[ray].t           = t;
    rays[ray].r           = r;
    rays[ray].theta       = theta;
    rays[ray].phi         = phi;
    rays[ray].pt          = pt;
    rays[ray].pr          = pr;
    rays[ray].ptheta      = ptheta;
    rays[ray].pphi        = pphi;
    rays[ray].rdot_sign   = rdot_sign;
    rays[ray].thetadot_sign = thetadot_sign;
    rays[ray].rdot_flips   = rdot_flips;
    rays[ray].equatorial_crossings = equatorial_crossings;

    if (steps > 0) rays[ray].steps += steps;
    if (rays[ray].status & RAY_STATUS_STEPLIM)
        rays[ray].steps = -rays[ray].steps;   // negative steps flags a stuck/failed ray
    return steps;
}

// =============================================================================
// Symplectic (Mino-time) integrator
// =============================================================================
//
// Integrates the canonical Mino-time system (see the Mino-time helpers in kerr.h):
//
//   H_M = 1/2 p_u^2 + 1/2 p_theta^2 + V_r(u) + V_theta(theta) = 0,   d lambda_M = d lambda / rhosq
//
// with r = 1 + b cosh(u), b = sqrt(1 - a^2).  Unlike the Euler/RK propagators the momenta are
// integrated directly (no square roots of R(r) and Theta(theta), so no sign-flip tracking), and
// the constants of motion k and h enter only through the potentials; Q is conserved by the
// integrator to within its truncation error and can be used as a diagnostic (carter_Q()).
//
// One step is a composition of Störmer-Verlet substeps kick(h/2) drift(h) kick(h/2):
//   kick  — momentum updates from dV_r/du and dW/dtheta, plus the t and phi quadratures
//   drift — free radial drift u += h p_u and the exact great-circle polar flow
//           (mino_polar_flow, which handles the h^2/sin^2theta barrier near the polar axis)
// Composition weights: order 2 = {1}; order 4 = Yoshida {w1, w0, w1}; order 6 = Yoshida
// solution A {w3, w2, w1, w0, w1, w2, w3} (Yoshida 1990, Phys. Lett. A 150, 262).
//
// Step size: fixed Mino-time step h0 (set_symplectic_step, default 1/precision) so that the
// integration is exactly symplectic in the strong-field region, capped far from the black hole
// by max_tstep / |dt/dlambda_M| and max_phistep / |dphi/dlambda_M| (at large r this is a constant
// affine step of max_tstep, which is needed because a fixed Mino step reaches r = infinity in
// finite Mino time).  The cap is applied at all radii (MAXDT_RLIM is not used here).
//
// Termination: when a full composed step crosses the theta limit or rlim, the step length that
// lands exactly on the boundary is found by bisection on the full composed step (so the ray
// finishes on the surface with consistent t, phi and momenta), then theta / r is snapped to the
// boundary value.  Horizon: u <= 0 or r <= horizon after a step.
//

template <typename T>
inline int Raytracer<T>::propagate_symplectic_impl(int ray, const T rlim, const T thetalim, RayDestination<T>* dest,
                                                    const int steplim, TextOutput* outfile, int write_step,
                                                    T write_rmax, T write_rmin, bool write_cartesian)
{
    //
    // Shared implementation of both propagate_symplectic() overloads.  If dest is non-null the ray is
    // stopped when dest->reached(r, theta, phi, theta_prev) becomes true and thetalim is ignored;
    // otherwise the fixed theta limit is used (thetalim > 0: stop at theta >= thetalim; thetalim < 0:
    // stop at theta <= |thetalim|; thetalim == 0: no theta limit).
    //
    int steps = 0;

    T x, y, z;

    const T a = spin;
    const T b = mino_b<T>(a);
    const T k = rays[ray].k;
    const T h = rays[ray].h;   // z-angular momentum constant of motion
    const T Q = rays[ray].Q;

    int rdot_flips           = rays[ray].rdot_flips;
    int equatorial_crossings = rays[ray].equatorial_crossings;

    // canonical Mino-time state
    typedef MinoState<T> State;
    State s;
    s.t   = rays[ray].t;
    s.phi = rays[ray].phi;
    s.theta = rays[ray].theta;
    mino_init<T>(rays[ray].r, rays[ray].theta, k, h, Q, rays[ray].rdot_sign, rays[ray].thetadot_sign, a,
                 s.u, s.pu, s.ptheta);
    T r, sqrt_delta;
    mino_r_from_u<T>(s.u, a, r, sqrt_delta);

    MinoStepper<T> stepper(a, k, h, symp_order);

    const T h0 = get_symplectic_step();
    // The max_tstep / max_phistep caps are only applied outside r_cap.  Close to the horizon
    // dt/dlambda_M diverges (t -> infinity as r -> r+), so a coordinate-time cap there would make
    // an infalling ray creep towards the horizon in ever smaller steps without ever reaching it;
    // with the fixed Mino step it simply overshoots to u <= 0 and is flagged.
    const T r_cap = 2 * horizon;

    // one full composed step of Mino time hstep (sets stepper.hit_horizon)
    auto composed_step = [&](State& q, T hstep) { stepper.step(q, hstep); };
    // boundary predicates on a trial state (theta_prev is theta at the start of the current step)
    auto theta_crossed = [&](const State& q, T theta_prev)
    {
        if (dest != nullptr)
            return dest->reached(1 + b*cosh(q.u), q.theta, q.phi, theta_prev);
        return (thetalim > 0 && q.theta >= thetalim) || (thetalim < 0 && q.theta <= abs(thetalim));
    };
    auto r_crossed = [&](const State& q)
    {
        return rlim > 0 && (1 + b*cosh(q.u)) >= rlim;
    };

    bool write_started = false;

    // --- Main integration loop ---
    while (r < rlim
           && (dest != nullptr || (thetalim > 0 && s.theta < thetalim) || (thetalim < 0 && s.theta > abs(thetalim)) || thetalim == 0)
           && steps < steplim)
    {
        ++steps;

        const T rhosq = r*r + (a*cos(s.theta))*(a*cos(s.theta));

        // --- step size: fixed Mino step, capped by the coordinate-time and phi steps ---
        // Far-field policy (mino_step_size in mino_stepper.h): inside maxtstep_rlim the coordinate-time
        // step is capped at max_tstep (a constant affine step at large r); beyond it the cap grows in
        // proportion to r, i.e. the radial step becomes a fixed fraction of r, as for the Euler/RK4 step
        // heuristic.  Pass a very large rlim to set_max_tstep() to keep the constant cap all the way out.
        T tdot, phidot;
        const T step = mino_step_size<T>(r, s.theta, k, h, a, h0, r_cap, max_tstep, maxtstep_rlim, max_phistep, tdot, phidot);

        // --- physicality flags (same criteria as the other propagators) ---
        {
            const T pt   = tdot / rhosq;
            const T pphi = phidot / rhosq;
            if (pt <= 0)
                rays[ray].status |= RAY_STATUS_ERGO;
            if ((1 - 2*r/rhosq)*pt + (2*a*r*sin(s.theta)*sin(s.theta)/rhosq)*pphi < 0)
                rays[ray].status |= RAY_STATUS_NEG_ENERGY;
        }

        // --- take the step ---
        const State s_prev = s;
        composed_step(s, step);

        // --- exact landing on the theta limit or rlim: bisection on the full composed step ---
        bool reached_dest = false, reached_rlim = false;
        if (!stepper.hit_horizon && (theta_crossed(s, s_prev.theta) || r_crossed(s)))
        {
            T lo = 0, hi = step;
            State trial;
            for (int it = 0; it < 60 && (hi - lo) > T(1e-13)*step; it++)
            {
                const T mid = (lo + hi) / 2;
                trial = s_prev;
                composed_step(trial, mid);
                if (stepper.hit_horizon || theta_crossed(trial, s_prev.theta) || r_crossed(trial)) hi = mid; else lo = mid;
            }
            s = s_prev;
            composed_step(s, hi);
            // snap onto the boundary that was reached (a fixed theta limit or rlim; a general
            // RayDestination surface is left where the bisection put it, within 1e-13 of a step)
            if (!stepper.hit_horizon)
            {
                if (theta_crossed(s, s_prev.theta))
                {
                    reached_dest = true;
                    if (dest == nullptr) s.theta = (thetalim > 0) ? thetalim : abs(thetalim);
                }
                else
                {
                    reached_rlim = true;
                    s.u = mino_u_from_r<T>(rlim, a);
                }
            }
        }

        // --- bookkeeping ---
        if (s_prev.pu * s.pu < 0) rdot_flips++;
        if ((s_prev.theta < M_PI_2 && s.theta >= M_PI_2) || (s_prev.theta > M_PI_2 && s.theta <= M_PI_2))
            ++equatorial_crossings;

        if (!isfinite(s.u) || !isfinite(s.theta) || !isfinite(s.pu) || !isfinite(s.ptheta))
        {
            // numerical breakdown: restore the last good state and flag the ray as failed
            s = s_prev;
            mino_r_from_u<T>(s.u, a, r, sqrt_delta);
            rays[ray].status |= RAY_STATUS_STEPLIM;
            break;
        }
        mino_r_from_u<T>(s.u, a, r, sqrt_delta);
        if (stepper.hit_horizon || r <= horizon)
        {
            rays[ray].status |= RAY_STATUS_HORIZON;
            break;
        }
        if (reached_rlim)
        {
            r = rlim;   // snapped onto the outer boundary (u -> r round-off could otherwise leave r just below rlim)
            break;
        }
        if (dest != nullptr && reached_dest)
        {
            rays[ray].status |= RAY_STATUS_DEST;
            break;
        }

        if (outfile != nullptr && (steps % write_step) == 0)
        {
            if ((write_rmax < 0 || r < write_rmax) && (write_rmin < 0 || r > write_rmin))
            {
                write_started = true;
                if (write_cartesian)
                {
                    cartesian<T>(x, y, z, r, s.theta, s.phi, a);
                    (*outfile) << s.t << x << y << z << endl;
                }
                else
                {
                    (*outfile) << s.t << r << s.theta << s.phi << endl;
                }
            }
            else if (write_started)
            {
                break;
            }
        }
    }  // end main loop

    if (steps >= steplim)
        rays[ray].status |= RAY_STATUS_STEPLIM;
    else if (r >= rlim)
        rays[ray].status |= RAY_STATUS_RLIM;
    else if (dest == nullptr && ((thetalim > 0 && s.theta >= thetalim) || (thetalim < 0 && s.theta <= abs(thetalim))))
        rays[ray].status |= RAY_STATUS_DEST;

    // --- write back in the Boyer-Lindquist / contravariant form used by the rest of the code ---
    T pt, pr, pth, pphi;
    mino_to_bl<T>(s.u, s.pu, s.theta, s.ptheta, k, h, a, r, pt, pr, pth, pphi);

    rays[ray].t      = s.t;
    rays[ray].r      = r;
    rays[ray].theta  = s.theta;
    rays[ray].phi    = s.phi;
    rays[ray].pt     = pt;
    rays[ray].pr     = pr;
    rays[ray].ptheta = pth;
    rays[ray].pphi   = pphi;
    rays[ray].rdot_sign     = (s.pu >= 0) ? 1 : -1;
    rays[ray].thetadot_sign = (s.ptheta > 0) ? 1 : -1;
    rays[ray].rdot_flips    = rdot_flips;
    rays[ray].equatorial_crossings = equatorial_crossings;

    if (steps > 0) rays[ray].steps += steps;
    if (rays[ray].status & RAY_STATUS_STEPLIM)
        rays[ray].steps = -rays[ray].steps;   // negative steps flags a stuck/failed ray
    return steps;
}

template <typename T>
inline int Raytracer<T>::propagate_symplectic(int ray, const T rlim, const T thetalim, const int steplim,
                                               TextOutput* outfile, int write_step,
                                               T write_rmax, T write_rmin, bool write_cartesian)
{
    return propagate_symplectic_impl(ray, rlim, thetalim, nullptr, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian);
}

template <typename T>
inline int Raytracer<T>::propagate_symplectic(int ray, const T rlim, RayDestination<T>* dest, const int steplim,
                                               TextOutput* outfile, int write_step,
                                               T write_rmax, T write_rmin, bool write_cartesian)
{
    return propagate_symplectic_impl(ray, rlim, T(0), dest, steplim, outfile, write_step, write_rmax, write_rmin, write_cartesian);
}

template class Raytracer<double>;
template class Raytracer<float>;
