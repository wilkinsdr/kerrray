/*
 * single_ray_bundle.h
 *
 *  Starts a five-ray bundle (caustic_bundle.h) at an arbitrary position on the observer's image plane, for
 *  the adaptive refinement of the plane and for the bisection of contour vertices in caustic_ent: five
 *  single-ray ImagePlane objects are constructed once (silently; the constructors print to cout) and
 *  re-initialised with ImagePlane::init_image_plane at each new position, so the initial data are exactly
 *  those of the regular image plane.  One instance per thread.
 */

#ifndef SINGLE_RAY_BUNDLE_H_
#define SINGLE_RAY_BUNDLE_H_

#include <iostream>
#include <streambuf>
#include "../raytracer/imageplane.h"
#include "caustic_bundle.h"

struct NullStreamBuffer : public std::streambuf { int overflow(int c) override { return c; } };

struct SingleRayBundle
{
    ImagePlane<double>* planes[5];
    double dist, incl, phi0, spin, delta, precision;

    // construct inside an omp critical section: cout is redirected while the planes are built
    SingleRayBundle(double dist_, double incl_, double phi0_, double spin_, double delta_, double precision_)
        : dist(dist_), incl(incl_), phi0(phi0_), spin(spin_), delta(delta_), precision(precision_)
    {
        NullStreamBuffer nb; std::streambuf* old = std::cout.rdbuf(&nb);
        for (int i = 0; i < 5; i++)
            planes[i] = new ImagePlane<double>(dist, incl, 0, 0, 1, 0, 0, 1, spin, phi0, precision);
        std::cout.rdbuf(old);
    }
    // destroy inside an omp critical section as well (the destructors print)
    ~SingleRayBundle()
    {
        NullStreamBuffer nb; std::streambuf* old = std::cout.rdbuf(&nb);
        for (int i = 0; i < 5; i++) delete planes[i];
        std::cout.rdbuf(old);
    }
    // initialise the bundle centred on image-plane position (x, y); returns false if undefined there
    // (the x = y = 0 ray).  emit receives the observer-frame energy of the centre ray (as redshift_start()).
    bool init(double x, double y, BundleRay b[5], double* emit = nullptr)
    {
        const double offx[5] = { 0, delta, -delta, 0, 0 }, offy[5] = { 0, 0, 0, delta, -delta };
        const Ray<double>* rays[5];
        for (int i = 0; i < 5; i++)
        {
            planes[i]->init_image_plane(dist, incl * M_PI / 180, phi0, x + offx[i], x + offx[i], 1,
                                        y + offy[i], y + offy[i], 1);
            rays[i] = planes[i]->rays;
        }
        if (emit != nullptr)
        {
            // observer-frame energy of the centre ray, as Raytracer::redshift_start(0, true) computes it
            // (ray_redshift with V = 0 in reverse mode returns recv / emit; with emit = 1 that is recv)
            const Ray<double>& c = rays[0][0];
            *emit = planes[0]->ray_redshift(0.0, true, false, c.r, c.theta, c.phi, c.k, c.h, c.Q,
                                            c.rdot_sign, c.thetadot_sign, 1.0);
        }
        return bundle_from_rays(rays, 0, -spin, b);
    }
};

#endif /* SINGLE_RAY_BUNDLE_H_ */
