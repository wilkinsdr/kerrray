/*
 * imagePlane.h
 *
 *  Created on: 25 Apr 2016
 *      Author: drw
 */

#ifndef IMAGEPLANE_H_
#define IMAGEPLANE_H_

#include "raytracer.h"

template <typename T>
class ImagePlane : public Raytracer<T>
{
private:
	T D;
	T incl;
	T phi0;

    T m_x0, m_xmax, m_dx;
    T m_y0, m_ymax, m_dy;

	int Nx, Ny;

public:
	ImagePlane( T dist, T inc, T x0, T xmax, T dx, T y0, T ymax, T dy, T spin, T phi, T precision = PRECISION);
	virtual ~ImagePlane() = default;   // required once this class has any virtual method (below); also
	                                    // incidentally makes it safe to hold a derived instance (e.g.
	                                    // LogImagePlane, log_imageplane.h) via unique_ptr<ImagePlane<T>>

	void init_image_plane(T D, T incl, T phi0, T x0, T xmax, T dx, T y0, T ymax, T dy);

	// Initial data of the ray that arrives at image-plane position (x, y) (rg), for a plane at distance D,
	// inclination incl (RADIANS) and azimuth phi0: position, momentum, constants of motion, direction signs,
	// alpha/beta.  init_image_plane() calls this for every pixel; applications that need rays at arbitrary
	// positions (the caustic bundle tracer) call it directly.  The two-argument form uses this plane's
	// geometry. virtual so a derived class (LogImagePlane) can be driven correctly through a
	// const ImagePlane<T>& reference (e.g. by RayTransfer, ray_transfer.h) -- without this, a call through
	// such a reference would always resolve to this class's own formula regardless of the referred object's
	// actual type, silently bypassing any override.
	virtual void init_ray(Ray<T>& ray, T x, T y, T D, T incl, T phi0) const;
	virtual void init_ray(Ray<T>& ray, T x, T y) const { init_ray(ray, x, y, D, incl * M_PI / 180, phi0); }
	T get_dist() const { return D; }
	T get_incl() const { return incl; }       // degrees
	T get_phi0() const { return phi0; }

	// Pixel-center grid query -- distinct from ray_x/ray_y/get_x_index/get_y_index below, which use this
	// class's original "grid point" (x0+i*dx, inclusive endpoints) convention and are unaffected by this.
	// virtual so LogImagePlane (log_imageplane.h) can override it with its own log-spaced grid; added so
	// RayTransfer (src/ray_transfer/ray_transfer.h) can query a plane's pixel grid uniformly regardless of
	// concrete type, without a separate adapter object.
	//
	// get_Nx()/get_Ny() deliberately do NOT return the private Nx/Ny members above: those are computed by
	// this class's constructor via its own "grid point" formula, Nx = ((xmax-x0)/dx)+1 (one MORE than the
	// pixel-center count, since it counts inclusive-endpoint points, not cells) -- reusing them here was
	// tried and found to be a real bug (traces one extra, out-of-range row/column, silently shifting every
	// aggregate sum by a few percent near a sharply-varying line, caught by the git-stash bit-identical
	// verification against ray_transfer_kerr_vs_flat's pre-existing output). The pixel-center count is
	// instead re-derived directly from (xmax-x0)/dx, rounded (not truncated: dx is itself
	// (xmax-x0)/N_original for whatever N the caller intended, so the round trip can land a ULP either side
	// of the integer -- round() recovers N_original exactly; truncation could silently be off by one).
	virtual int get_Nx() const { return (int)std::round((m_xmax - m_x0) / m_dx); }
	virtual int get_Ny() const { return (int)std::round((m_ymax - m_y0) / m_dy); }
	virtual T pixel_x(int ix) const { return m_x0 + (ix + T(0.5)) * m_dx; }
	virtual T pixel_y(int iy) const { return m_y0 + (iy + T(0.5)) * m_dy; }
	virtual T pixel_dx(int ix) const { return m_dx; }
	virtual T pixel_dy(int iy) const { return m_dy; }
	// Multiplicative factor for a pixel's contribution when summing into an aggregate over the whole grid
	// (e.g. RayTransfer::run_raytrace's spec_total/continuum_total). Fixed at 1, not pixel_dx*pixel_dy, to
	// match every existing RayTransfer consumer's longstanding unweighted-sum convention over this uniform
	// grid (every pixel here is the same size, so this is a deliberate historical-compatibility constant,
	// not a physically-motivated default for this class in general -- LogImagePlane's override is the
	// physically real one, a true non-uniform area).
	virtual T pixel_weight(int, int) const { return T(1); }

	void redshift_start( );
	void redshift(bool projradius);
	using Raytracer<T>::redshift;

    inline int get_x_index(int ix)
    {
        //
        // returns the index along the X direction of the 2D image place given the index of the 1D array element in raytracer variables
        //
        return static_cast<int>( ix / Ny );
    }

    inline int get_y_index(int ix)
    {
        //
        // returns the index along the Y direction of the 2D image place given the index of the 1D array element in raytracer variables
        //
        return ix % Ny;
    }

    inline T ray_x(int ix)
    {
        return m_x0 + get_x_index(ix) * m_dx;
    }

    inline T ray_y(int ix)
    {
        return m_y0 + get_y_index(ix) * m_dy;
    }
};

#endif /* IMAGEPLANE_H_ */
