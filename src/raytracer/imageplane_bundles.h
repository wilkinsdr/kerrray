/*
 * imageplane_bundles.h
 *
 * ImagePlaneBundles<T> — image plane that traces 5-ray bundles for accurate
 * local Jacobian computation.
 *
 * For each (ix, iy) grid point a bundle of 5 rays is traced:
 *
 *   base+0  centre  (x_i,       y_j      )
 *   base+1  east    (x_i+eps_x, y_j      )
 *   base+2  west    (x_i-eps_x, y_j      )
 *   base+3  north   (x_i,       y_j+eps_y)
 *   base+4  south   (x_i,       y_j-eps_y)
 *
 * where eps_x = eps_frac * dx  and  eps_y = eps_frac * dy.
 *
 * The Jacobian at each pixel is then computed from the satellite hit
 * positions using central finite differences over the small step eps,
 * rather than across the full inter-pixel spacing.  This gives much more
 * accurate local derivatives and avoids the neighbour-lookup failure that
 * marks pixels as SENTINEL whenever a grid neighbour lands in a different
 * image-order shell.
 *
 * Usage mirrors ImagePlane:
 *   ImagePlaneBundles<double> src(dist, incl, x0, xmax, dx, y0, ymax, dy,
 *                                  spin, phi0, precision, eps_frac);
 *   src.set_rk45_tol(tol);
 *   src.redshift_start();
 *   src.run_raytrace(&dest, integrator, r_max, show_progress);
 *   src.redshift(&dest, true);
 *   // Access centre / satellite ray indices:
 *   int c = src.centre_ray(ix, iy);
 *   auto& ray = src.rays[c];
 */

#ifndef IMAGEPLANE_BUNDLES_H_
#define IMAGEPLANE_BUNDLES_H_

#include <cmath>
#include <iostream>
#include "raytracer.h"
#include "ray_destination.h"

template <typename T>
class ImagePlaneBundles : public Raytracer<T>
{
public:
    static constexpr int RAYS_PER_BUNDLE = 5;

    int Nx, Ny;       // grid dimensions (fencepost: Nx+1 points along each axis)
    T   eps_x, eps_y; // bundle satellite offsets (rg)

private:
    T D, incl, phi0;
    T m_x0, m_dx;
    T m_y0, m_dy;

    // Initialise one ray at image-plane position (x, y).
    // Replicates init_image_plane() logic from ImagePlane exactly.
    void init_ray(int ix, T x, T y)
    {
        const T a = Raytracer<T>::spin;

        const T r     = sqrt(D*D + x*x + y*y);
        const T theta = acos((D*cos(incl) + y*sin(incl)) / r);
        const T phi   = phi0 + atan2(x, D*sin(incl) - y*cos(incl));

        const T pr     = D / r;
        const T ptheta = sin(acos(D / r)) / r;
        const T pphi   = x * sin(incl) /
                         (x*x + (D*sin(incl) - y*cos(incl)) *
                                (D*sin(incl) - y*cos(incl)));

        // Kerr metric coefficients
        const T rhosq   = r*r + (a*cos(theta)) * (a*cos(theta));
        const T delta   = r*r - 2*r + a*a;
        const T sigmasq = (r*r + a*a)*(r*r + a*a)
                          - a*a*delta*sin(theta)*sin(theta);

        const T e2nu  = rhosq * delta / sigmasq;
        const T e2psi = sigmasq * sin(theta)*sin(theta) / rhosq;
        const T omega = 2*a*r / sigmasq;

        const T g00 = e2nu - omega*omega*e2psi;
        const T g03 = omega * e2psi;
        const T g11 = -rhosq / delta;
        const T g22 = -rhosq;
        const T g33 = -e2psi;

        // Solve null-geodesic quadratic for p_t
        const T A = g00;
        const T B = 2*g03*pphi;
        const T C = g11*pr*pr + g22*ptheta*ptheta + g33*pphi*pphi;
        T pt = (-B + sqrt(B*B - 4*A*C)) / (2*A);
        if (pt < 0) pt = (-B - sqrt(B*B - 4*A*C)) / (2*A);

        Raytracer<T>::rays[ix].t      = 0;
        Raytracer<T>::rays[ix].r      = r;
        Raytracer<T>::rays[ix].theta  = theta;
        Raytracer<T>::rays[ix].phi    = phi;
        Raytracer<T>::rays[ix].pt     = pt;
        Raytracer<T>::rays[ix].pr     = pr;
        Raytracer<T>::rays[ix].ptheta = ptheta;
        Raytracer<T>::rays[ix].pphi   = pphi;

        Raytracer<T>::calculate_constants_from_p(ix, pt, pr, ptheta, pphi);
        Raytracer<T>::rays[ix].rdot_sign    = -1;
        Raytracer<T>::rays[ix].thetadot_sign = 1;
        Raytracer<T>::rays[ix].k            = 1;

        // Angular momentum constants (same analytic formula as ImagePlane)
        const T b    = sqrt(x*x + y*y);
        T       beta_ang = (b > 0) ? asin(y / b) : 0;
        if (x < 0) beta_ang = M_PI - beta_ang;

        const T h      = -b * sin(incl) * cos(beta_ang);
        const T ltheta =  b * sin(beta_ang);
        const T Q = ltheta*ltheta
                    - (a*cos(theta))*(a*cos(theta))
                    + (h / tan(theta)) * (h / tan(theta));

        Raytracer<T>::rays[ix].h             = h;
        Raytracer<T>::rays[ix].Q             = Q;
        Raytracer<T>::rays[ix].thetadot_sign = (ltheta >= 0) ? 1 : -1;
        Raytracer<T>::rays[ix].steps         = 0;
        Raytracer<T>::rays[ix].alpha         = x;
        Raytracer<T>::rays[ix].beta          = y;
    }

public:
    // Constructor.
    //   dist      — observer distance (rg)
    //   inc_deg   — observer inclination (degrees; converted to radians internally)
    //   x0,xmax,dx — image-plane X axis
    //   y0,ymax,dy — image-plane Y axis
    //   spin      — black hole spin a/M (negated internally for backward tracing)
    //   phi       — observer azimuthal angle (radians)
    //   precision — ray-stepper precision (passed to Raytracer)
    //   eps_frac  — satellite offset as fraction of pixel spacing (default 0.01)
    ImagePlaneBundles(T dist, T inc_deg,
                      T x0, T xmax, T dx,
                      T y0, T ymax, T dy,
                      T spin, T phi,
                      T precision = PRECISION,
                      T eps_frac  = 0.01)
        : Raytracer<T>( int((((xmax-x0)/dx) + 1) * (((ymax-y0)/dy) + 1))
                        * RAYS_PER_BUNDLE,
                        -1 * spin, precision ),
          Nx( int(((xmax-x0)/dx) + 1) ),
          Ny( int(((ymax-y0)/dy) + 1) ),
          eps_x(eps_frac * dx),
          eps_y(eps_frac * dy),
          D(dist),
          incl(inc_deg * M_PI / 180.0),
          phi0(phi),
          m_x0(x0), m_dx(dx),
          m_y0(y0), m_dy(dy)
    {
        std::cout << "Setting up image plane bundles with ("
                  << Nx << 'x' << Ny << ") bundle centres, "
                  << RAYS_PER_BUNDLE << " rays each = "
                  << Nx * Ny * RAYS_PER_BUNDLE << " total rays" << std::endl;
        std::cout << "  bundle eps_x=" << eps_x << " eps_y=" << eps_y
                  << "  (eps_frac=" << eps_frac << ")" << std::endl;

        for (int i = 0; i < Nx; i++)
        {
            // NOTE: matches ImagePlane's i*dy convention (dy used for both axes)
            const T x = m_x0 + i * m_dy;
            for (int j = 0; j < Ny; j++)
            {
                const T y    = m_y0 + j * m_dy;
                const int base = (i * Ny + j) * RAYS_PER_BUNDLE;

                init_ray(base + 0, x,         y        );   // centre
                init_ray(base + 1, x + eps_x, y        );   // east
                init_ray(base + 2, x - eps_x, y        );   // west
                init_ray(base + 3, x,         y + eps_y);   // north
                init_ray(base + 4, x,         y - eps_y);   // south
            }
        }
    }

    // ---- Bundle ray index accessors ----
    inline int centre_ray(int ix, int iy) const { return (ix*Ny + iy)*RAYS_PER_BUNDLE + 0; }
    inline int east_ray  (int ix, int iy) const { return (ix*Ny + iy)*RAYS_PER_BUNDLE + 1; }
    inline int west_ray  (int ix, int iy) const { return (ix*Ny + iy)*RAYS_PER_BUNDLE + 2; }
    inline int north_ray (int ix, int iy) const { return (ix*Ny + iy)*RAYS_PER_BUNDLE + 3; }
    inline int south_ray (int ix, int iy) const { return (ix*Ny + iy)*RAYS_PER_BUNDLE + 4; }

    // ---- Redshift wrappers (same semantics as ImagePlane) ----
    void redshift_start()
    {
        Raytracer<T>::redshift_start(0, true);
    }

    // Make all Raytracer redshift() overloads visible (the RayDestination
    // overload is the one caustic programs use).
    using Raytracer<T>::redshift;
};

#endif /* IMAGEPLANE_BUNDLES_H_ */
