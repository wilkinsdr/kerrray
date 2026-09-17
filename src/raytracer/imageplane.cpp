/*
 * imagePlane.cpp
 *
 *  Created on: 25 Apr 2016
 *      Author: drw
 */

#include "imageplane.h"

template <typename T>
ImagePlane<T>::ImagePlane( T dist, T inc, T x0, T xmax, T dx, T y0, T ymax, T dy, T spin, T phi, T precision )
	: Raytracer<T>( (((xmax - x0) / dx) + 1) * (((ymax - y0) / dy) + 1) , -1 * spin , precision ),
	        Nx(((xmax - x0) / dx) + 1),
	        Ny(((ymax - y0) / dy) + 1),
	        D(dist),
	        incl(inc),
	        phi0(phi),
            m_x0(x0), m_xmax(xmax), m_dx(dx),
            m_y0(y0), m_ymax(ymax), m_dy(dy)
{
	cout << "Setting up image plane with (" << Nx << 'x' << Ny << ") rays" << endl;
	init_image_plane(D, incl * M_PI / 180, phi0, x0, xmax, dx, y0, ymax, dy);
}

//template <typename T>
//ImagePlane<T>::~ImagePlane()
//{
//
//}

template <typename T>
void ImagePlane<T>::init_image_plane(T D, T incl, T phi0,
									T x0, T xmax, T dx,
                                    T y0, T ymax, T dy)
{
	const int Nx = ((xmax - x0) / dx) + 1;
	const int Ny = ((ymax - y0) / dy) + 1;
	
	const double a = Raytracer<T>::spin;
	
	for(int i=0; i<Nx; i++)
	{
		const T x = x0 + i * dx;

		for (int j = 0; j < Ny; j++)
		{
			const int ix = i * Ny + j;

			const T y = y0 + j*dy;

			// initialise position of photon
			const T t = 0;
			const T r = sqrt(D*D + x*x + y*y);
			const T theta = acos( (D*cos(incl) + y*sin(incl)) / r );
			const T phi = phi0 + atan2( x, D*sin(incl) - y*cos(incl) );
			
			// and the momentum
			const T pr = D/r;
			const T ptheta = sin(acos(D/r))/r;
			const T pphi = x*sin(incl)/(x*x+(D*sin(incl)-y*cos(incl))*(D*sin(incl)-y*cos(incl)));

			// metric coefficients
			const T rhosq = r*r + (a*cos(theta))*(a*cos(theta));
			const T delta = r*r - 2*r + a*a;
			const T sigmasq = (r*r + a*a)*(r*r + a*a) - a*a*delta*sin(theta)*sin(theta);

			const T e2nu = rhosq * delta / sigmasq;
			const T e2psi = sigmasq * sin(theta)*sin(theta) / rhosq;
			const T omega = 2*a*r / sigmasq;

			const T g00 = e2nu - omega*omega*e2psi;
			const T g03 = omega*e2psi;
			const T g11 = -rhosq/delta;
			const T g22 = -rhosq;
			const T g33 = -e2psi;

			// solve quadratic in pt to make this a null vector
			const T A = g00;
			const T B = 2*g03*pphi;
			const T C = g11*pr*pr + g22*ptheta*ptheta + g33*pphi*pphi;

			// take the appropriate (positive) root
			T pt = ( -B + sqrt(B*B - 4*A*C) ) / (2*A);
			if(pt < 0) pt = ( -B - sqrt(B*B - 4*A*C) ) / (2*A);

			Raytracer<T>::rays[ix].t = t;
			Raytracer<T>::rays[ix].r = r;
			Raytracer<T>::rays[ix].theta = theta;
			Raytracer<T>::rays[ix].phi = phi;
			Raytracer<T>::rays[ix].pt = pt;
			Raytracer<T>::rays[ix].pr = pr;
			Raytracer<T>::rays[ix].ptheta = ptheta;
			Raytracer<T>::rays[ix].pphi = pphi;

			// calculate constants of motion
			Raytracer<T>::calculate_constants_from_p(ix, pt, pr, ptheta, pphi);
			Raytracer<T>::rays[ix].rdot_sign = -1;
			Raytracer<T>::rays[ix].thetadot_sign = 1;

			Raytracer<T>::rays[ix].k = 1;

			const T b = sqrt(x*x + y*y);
			T beta = asin(y/b);
			if(x < 0) beta=M_PI-beta;

			// Constants of motion of the photon that arrives at the observer at image-plane position
			// (x, y) = (b cos beta, b sin beta) (Cunningham & Bardeen 1973: alpha = -h / sin(incl) = x,
			// beta = p_theta at the observer = y), for the ray traced BACKWARDS from the plane.
			// The backward trace runs in the spin-reversed spacetime (Raytracer constructed with -spin), in
			// which the reversed ray has h -> -h_phys = +x sin(incl) and p_theta -> -p_theta,phys, i.e. a ray
			// above the plane centre (y > 0) initially moves towards smaller theta.  (The earlier convention,
			// h = -x sin(incl) with theta increasing for y > 0, sent the ray from plane position (x, y) past the
			// black hole on the (-x, -y) side, so that the whole family crossed the line of sight at a distance
			// D/2 from the black hole; imaging applications compensated with a flip_image option.)
			T h = b*sin(incl)*cos(beta);
			T ltheta = b*sin(beta);
			T Q = (ltheta*ltheta) - (a*cos(theta))*(a*cos(theta))+((h/tan(theta)))*((h/tan(theta)));

			Raytracer<T>::rays[ix].h = h;
			Raytracer<T>::rays[ix].Q = Q;

			Raytracer<T>::rays[ix].thetadot_sign = (ltheta>=0) ? -1 : 1;
			
			Raytracer<T>::rays[ix].steps = 0;

            Raytracer<T>::rays[ix].alpha = x;
            Raytracer<T>::rays[ix].beta = y;
		}
	}
}

template <typename T>
void ImagePlane<T>::redshift_start( )
{
	//
	// Call the redshift_start function of the base class using the source's angular velocity
	//
    Raytracer<T>::redshift_start(0, true);
}

template <typename T>
void ImagePlane<T>::redshift(bool projradius )
{
	//
	// Call the redshift_start function of the base class using the angular velocity for a circular orbit at the ray's end point
	// for rays incident on the accretion disc
	//
	Raytracer<T>::redshift(-1, true, projradius);
}

template class ImagePlane<double>;
template class ImagePlane<float>;
