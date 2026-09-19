/*
 * gpu_raytracer.h
 *
 *  Created on: 4 Sep 2013
 *      Author: drw
 *
 *  General relativistic ray tracing in the Kerr spacetime
 *
 *  This class provides the basic functionality fundamental to the rays (e.g. propagation, redshift calculation)
 *  and is used as the base class for specific types of source for ray tracing in different scenarios.
 *
 *  This class is a template to set the floating point precision (e.g. float or double)
 */

#ifndef RAYTRACER_H_
#define RAYTRACER_H_

// set a default precision used for variable integration step sizes based on distance from event horizon
#define PRECISION 100
#define TOL 100
//
#define THETA_PRECISION 50
// default maximum allowed step in co-ordinate time
#define MAXDT 1
// only obey this inside this radius so rays don't take ages a long way from the black hole
#define MAXDT_RLIM 100
// default maximum allowed step in phi
#define MAXDPHI 0.1
// set a default outer radius for ray propagation
#define RLIM 1000
// total number of steps allowed per ray before it's aborted
#define STEPLIM 10000000
// step limit for the RK45/DOPRI5 adaptive integrator.  Adaptive steps are
// larger than fixed-step integrators, so legitimate rays finish in far fewer
// steps.  Empirically (spin=0.998, lamppost r=5): well-behaved rays use
// ≤34K steps; stuck photon-sphere orbiting rays never converge.  100K gives
// ~3x headroom above the observed maximum while terminating stuck rays
// 100x faster than STEPLIM.
#define RK45_STEPLIM 100000
// step limit for the symplectic (Mino-time) integrator.  Its step is fixed in Mino time near
// the black hole and capped to a fixed affine step far away (via max_tstep), so the count is
// bounded by ~r_max/max_tstep plus the strong-field orbits; 1M gives ample headroom for
// photon-sphere rays while still terminating stuck ones.
#define SYMP_STEPLIM 1000000
// horizon threshold for the symplectic integrator, in the radial coordinate u (r = 1 + b cosh u,
// so u = 0 is the horizon and u = 0.05 is r - r+ ~ 1e-3 b).  The radial potential is singular at
// u = 0 and photon turning points never lie below u ~ 0.5 (the circular photon orbit) for any spin,
// so a ray reaching u < SYMP_U_HORIZON is unambiguously captured.
#define SYMP_U_HORIZON 0.05
// number of integration steps per GPU thread before integration is paused and kernel must be called again
// to avoid thread time limits on GPUs running X servers
#define THREAD_STEPLIM 10000000
// smallest integration step  to prevent infinite loops of infinitesimal steps
#define MIN_STEP 1E-3
// minimum number of integration steps before sign of rdot and thetadot are allowed to change
#define COUNT_MIN 100

#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#include "../include/kerr.h"
#include "../include/text_output.h"
#include "../include/progress_bar.h"

// Ray propagation status bit flags (can be combined with bitwise OR)
constexpr int RAY_STATUS_DEST       = (1 << 0);  // ray reached destination (user-defined, or polar angle limit in thetalim overloads)
constexpr int RAY_STATUS_HORIZON    = (1 << 1);  // ray fell through event horizon
constexpr int RAY_STATUS_RLIM       = (1 << 2);  // ray reached outer radial limit
constexpr int RAY_STATUS_STEPLIM    = (1 << 3);  // ray exceeded maximum step count
constexpr int RAY_STATUS_ERGO       = (1 << 4);  // ray entered ergosphere (non-physical, pt <= 0)
constexpr int RAY_STATUS_NEG_ENERGY = (1 << 5);  // dot product of 4-momentum with timelike Killing vector is negative (non-physical)

template <typename T>
struct Ray
{
    T t, r, theta, phi;
    T pt, pr, ptheta, pphi;
    T k, h, Q;
    T emit, redshift;
    int steps;
    int status;
    int rdot_sign, thetadot_sign;
    int rdot_flips;           // number of radial turning points (sign flips of pr) encountered during propagation
    int equatorial_crossings; // number of times theta crossed pi/2; image order for photon rings = equatorial_crossings - 1
    T alpha, beta;
};

template <typename T> class RayDestination;

/// Selects the integration algorithm used by run_raytrace().
///   Euler      — fixed-step, semi-analytic (momenta from constants of motion)
///   RK4        — classical 4th-order Runge-Kutta, fixed step
///   RK45       — adaptive Dormand-Prince (DOPRI5)
///   Symplectic — Mino-time symplectic integrator (Störmer-Verlet / Yoshida composition);
///                integrates the canonical momenta directly, order set by set_symplectic_order()
enum class Integrator { Euler, RK4, RK45, Symplectic };

template <typename T>
class Raytracer
{
private:
    T precision;
    T theta_precision;
    T max_tstep;
    T max_phistep;
    T maxtstep_rlim;
    T rk45_tol;          // per-step mixed abs/rel error tolerance for the DOPRI5 adaptive controller
    T symp_step;         // Mino-time step h0 for the symplectic integrator (<= 0: use 1/precision)
    int symp_order;      // composition order for the symplectic integrator: 2 (Verlet), 4 or 6 (Yoshida, default 6)

protected:	// these members need to be accessible by derived classes to set up different X-ray sources
	int nRays;
	T spin;
	T horizon;

	inline void calculate_constants(int ray, T alpha, T beta, T V, T E);
	inline void calculate_constants_from_p(int ray, T pt, T pr, T ptheta, T pphi);
	inline void calculate_constants_from_p(Ray<T>& ray, T pt, T pr, T ptheta, T pphi) const;

    // shared implementation of the two propagate_symplectic() overloads (dest == nullptr: use thetalim)
    inline int propagate_symplectic_impl(int ray, const T rlim, const T thetalim, RayDestination<T>* dest, const int steplim,
                                         TextOutput* outfile, int write_step, T write_rmax, T write_rmin, bool write_cartesian);

public:
    Ray<T> *rays;

    Raytracer( int num_rays, T spin, T precision = PRECISION, T init_max_phistep = MAXDPHI, T init_max_tstep = MAXDT );
    ~Raytracer( );

    // steplim: maximum integration steps per ray.  Pass -1 (default) to use the
    // integrator's built-in limit (STEPLIM for Euler/RK4, RK45_STEPLIM for RK45).
    void run_raytrace(Integrator method = Integrator::Euler, T theta_max = M_PI / 2, T r_max = 1000,
                      int show_progress = 1, TextOutput* outfile = 0,
                      int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true,
                      int steplim = -1);
    inline int propagate(int ray, const T rlim, const T thetalim, const int steplim, TextOutput* outfile = 0
                         , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);

    void run_raytrace(RayDestination<T>* dest, Integrator method = Integrator::Euler, T r_max = 1000,
                      int show_progress = 1, TextOutput* outfile = 0,
                      int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true,
                      int steplim = -1);
    inline int propagate_rk4(int ray, const T rlim, const T thetalim, const int steplim, TextOutput* outfile = 0
                              , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);
    inline int propagate_rk4(int ray, const T rlim, RayDestination<T>* dest, const int steplim, TextOutput* outfile = 0
                              , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);
    inline int propagate_rk45(int ray, const T rlim, const T thetalim, const int steplim, TextOutput* outfile = 0
                               , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);
    inline int propagate_rk45(int ray, const T rlim, RayDestination<T>* dest, const int steplim, TextOutput* outfile = 0
                               , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);
    inline int propagate_symplectic(int ray, const T rlim, const T thetalim, const int steplim, TextOutput* outfile = 0
                               , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);
    inline int propagate_symplectic(int ray, const T rlim, RayDestination<T>* dest, const int steplim, TextOutput* outfile = 0
                               , int write_step = 1, T write_rmax = -1, T write_rmin = -1, bool write_cartesian = true);

    void redshift_start(T V, bool reverse = false, bool projradius = false);
    // the energy of one ray in the frame of an emitter/observer with angular velocity V at the ray's current
    // position (what redshift_start() stores in Ray::emit for every ray)
    inline T emit_energy(const Ray<T>& ray, T V, bool reverse = false, bool projradius = false) const;
    void redshift(T V, bool reverse = false, bool projradius = false, int motion = 0);
    void redshift(RayDestination<T>* dest, bool reverse = false, bool projradius = false, int motion = 0);
	inline T ray_redshift( T V, bool reverse, bool projradius, T r, T theta, T phi, T k, T h, T Q, int rdot_sign, int thetadot_sign, T emit, int motion = 0 ) const;
    inline T ray_redshift( const T et[4], bool reverse, T r, T theta, T phi, T k, T h, T Q, int rdot_sign, int thetadot_sign, T emit ) const;

    void range_phi(T min = -1 * M_PI, T max = M_PI);

    void calculate_momentum( );

    int get_count( )
    {
    	//
    	// Returns the total number of threads (number of entries in raytrace variable arrays).
    	// This is not necessarily the number of rays as those out of the specified bounds will not have been traced,
    	// though this should be used for iterating over the raytrace results.
    	//
    	return nRays;
    }

    void set_boundary(T r = -1)
    {
    	//
		// Sets a hard boundary (inner radius) through which rays cannot pass, e.g. for raytracing around neutron stars
    	// If called with no argument, sets the boundary to the event horizon.
		//
    	if(r > 0)
    		horizon = r;
    	else
    		horizon = kerr_horizon<T>(spin);
    }

    T calculate_horizon( )
    {
    	return kerr_horizon<T>(spin);
    }

    void set_precision(T precision)
    {
        precision = precision;
    }

    void set_precision(T precision = PRECISION, T theta_precision = THETA_PRECISION)
    {
        precision = precision;
        theta_precision = theta_precision;
    }

    // Set the per-step error tolerance for the DOPRI5 adaptive step controller.
    // Smaller values force tighter accuracy and more steps; default is 1e-8.
    // Useful for reducing photon-sphere separatrix sensitivity at the cost of
    // longer run times.
    void set_rk45_tol(T tol) { rk45_tol = tol; }
    T    get_rk45_tol() const { return rk45_tol; }

    // Symplectic integrator controls.  The step is the Mino-time step h0 used in the strong-field
    // region (the affine step is rho^2 * h0); far from the black hole the step is additionally capped
    // by max_tstep / max_phistep expressed in Mino time.  A non-positive step means 1/precision.
    // The order selects the composition: 2 = Störmer-Verlet, 4 = Yoshida 4th order, 6 = Yoshida 6th
    // order (default: the work-precision sweep in docs/plan_symplectic_integrator.md shows order 6 at
    // max_tstep = 1 to be both faster and more accurate than order 4 at max_tstep = 0.25).
    void set_symplectic_step(T h0) { symp_step = h0; }
    T    get_symplectic_step() const { return (symp_step > 0) ? symp_step : T(1) / precision; }
    void set_symplectic_order(int order) { symp_order = (order == 2 || order == 4) ? order : 6; }
    int  get_symplectic_order() const { return symp_order; }

    void set_max_tstep(T max, T rlim = MAXDT_RLIM)
    {
        max_tstep = max;
        maxtstep_rlim = rlim;
    }

    void set_max_phistep(T max)
    {
        max_phistep = max;
    }

};

#endif /* RAYTRACER_H_ */
