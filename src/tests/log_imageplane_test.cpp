//
// log_imageplane_test.cpp
//
// Self-checking tests for LogImagePlane (raytracer/log_imageplane.h):
//   (1) off-origin, LogImagePlane::init_ray reproduces ImagePlane::init_ray exactly (the delegation itself)
//   (2) the per-ray area weights sum to the full bounding-box area (grid tiles with no gap/overlap)
//   (3) grid coordinates are strictly ascending, with the expected sign/range
//   (4) a grid forced to place a ray exactly on the optical axis (x=y=0) produces no NaN/Inf anywhere
//   (5) the inherited Raytracer machinery (run_raytrace/redshift_start/redshift) works unmodified
//
// Build (from the project root, no cfitsio needed):
//   g++ -O2 -std=c++17 -fopenmp -I src src/tests/log_imageplane_test.cpp src/raytracer/raytracer.cpp \
//       src/raytracer/imageplane.cpp src/raytracer/log_imageplane.cpp -o bin/log_imageplane_test
//
#include <iostream>
#include <cmath>

#include "raytracer/imageplane.h"
#include "raytracer/log_imageplane.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& msg)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << msg << endl;
    if (!ok) ++failures;
}

int main()
{
    const double spin = 0.9, incl = 60, dist = 1000, phi0 = 0;

    // Deliberately asymmetric x/y grid parameters throughout: catches an x/y-swapped weight or index bug
    // that a symmetric test configuration (x_min==y_min, Nx==Ny, ...) would not expose.
    const double x_min = 2, x_max = 200; const int Nx = 30, Nlinx = 4;
    const double y_min = 3, y_max = 100; const int Ny = 20, Nliny = 6;

    LogImagePlane<double> plane(dist, incl, x_min, x_max, Nx, Nlinx, y_min, y_max, Ny, Nliny, spin, phi0);
    const int n = plane.get_count();

    cout << "\n(1) init_ray reproduces ImagePlane::init_ray away from the origin" << endl;
    {
        ImagePlane<double> ref(dist, incl, 1, 2, 1, 1, 2, 1, spin, phi0);   // any instance; init_ray is const, arg-only
        const double incl_rad = incl * M_PI / 180;
        double xs[] = {-150, -37.2, -x_min, x_min, 5.5, 80, x_max};
        double ys[] = {-60, -y_min, y_min, 1.3, 40, y_max};
        double mx = 0;
        int npairs = 0;
        for (double x : xs)
        {
            for (double y : ys)
            {
                Ray<double> r1, r2;
                ref.init_ray(r1, x, y, dist, incl_rad, phi0);
                plane.init_ray(r2, x, y, dist, incl_rad, phi0);
                double d[] = { fabs(r1.r-r2.r), fabs(r1.theta-r2.theta), fabs(r1.phi-r2.phi), fabs(r1.k-r2.k),
                               fabs(r1.h-r2.h), fabs(r1.Q-r2.Q), fabs(r1.pt-r2.pt), fabs(r1.pr-r2.pr),
                               fabs(r1.ptheta-r2.ptheta), fabs(r1.pphi-r2.pphi),
                               fabs(r1.alpha-r2.alpha), fabs(r1.beta-r2.beta) };
                for (double v : d) if (v > mx) mx = v;
                if (r1.rdot_sign != r2.rdot_sign || r1.thetadot_sign != r2.thetadot_sign)
                    mx = 1e99;
                ++npairs;
            }
        }
        cout << "  " << npairs << " (x,y) pairs: max |diff| = " << scientific << mx << fixed << endl;
        check(mx == 0, "(1) LogImagePlane::init_ray == ImagePlane::init_ray away from the origin");
    }

    cout << "\n(2) grid coordinates: ascending, correct sign/range" << endl;
    {
        bool ok = true;
        for (int i = 1; i < plane.get_Nx(); i++)
        {
            const int ix_prev = (i - 1) * plane.get_Ny();
            const int ix = i * plane.get_Ny();
            if (!(plane.ray_x(ix) > plane.ray_x(ix_prev))) ok = false;
        }
        for (int j = 1; j < plane.get_Ny(); j++)
        {
            if (!(plane.ray_y(j) > plane.ray_y(j - 1))) ok = false;
        }
        // First/last column should sit just inside +/- x_max (geometric-mean bin centers, strictly < x_max
        // in magnitude); first column negative, last column positive.
        const double x_first = plane.ray_x(0);
        const double x_last  = plane.ray_x((plane.get_Nx() - 1) * plane.get_Ny());
        if (!(x_first < -x_min && x_first > -x_max)) ok = false;
        if (!(x_last  >  x_min && x_last  <  x_max)) ok = false;
        cout << "  Nx=" << plane.get_Nx() << " Ny=" << plane.get_Ny()
             << ", x range [" << x_first << ", " << x_last << "]" << endl;
        check(ok, "(2) grid coordinates strictly ascending with expected sign/range");
    }

    cout << "\n(3) weight-sum identity (exact bounding-box area, no gap/overlap)" << endl;
    {
        double total_weight = 0;
        for (int ix = 0; ix < n; ix++) total_weight += plane.ray_weight(ix);
        const double expected = (2 * x_max) * (2 * y_max);
        const double relerr = fabs(total_weight - expected) / expected;
        cout << "  sum(weight) = " << total_weight << ", expected " << expected
             << ", relative error = " << scientific << relerr << fixed << endl;
        check(relerr < 1e-9, "(3) sum of ray_weight() over the whole grid == (2*x_max)*(2*y_max)");
    }

    cout << "\n(4) origin safety: a ray forced exactly onto x=y=0 stays finite" << endl;
    {
        // Nlinx=Nliny=1: the single linear-zone bin's center is exactly (0.5*2*lo - lo) == 0 in floating
        // point for any lo (multiplying/dividing by 2 is exact in IEEE754), so this grid is guaranteed to
        // place a ray exactly at the optical axis.
        LogImagePlane<double> origin_plane(dist, incl, 2.0, 50.0, 5, 1, 3.0, 40.0, 4, 1, spin, phi0);
        bool all_finite = true;
        for (int ix = 0; ix < origin_plane.get_count(); ix++)
        {
            const Ray<double>& r = origin_plane.rays[ix];
            double f[] = {r.r, r.theta, r.phi, r.pt, r.pr, r.ptheta, r.pphi, r.h, r.Q, r.k};
            for (double v : f) if (!std::isfinite(v)) all_finite = false;
        }
        // Confirm the test actually exercised the degenerate point, not just finiteness in general.
        bool hit_origin = false;
        for (int ix = 0; ix < origin_plane.get_count(); ix++)
            if (origin_plane.ray_x(ix) == 0.0 && origin_plane.ray_y(ix) == 0.0) hit_origin = true;
        cout << "  " << origin_plane.get_count() << " rays, origin ray present: " << (hit_origin ? "yes" : "no")
             << ", all finite: " << (all_finite ? "yes" : "no") << endl;
        check(hit_origin, "(4) test grid genuinely places a ray at x=y=0");
        check(all_finite, "(4) every field of every ray is finite (origin guard works)");
    }

    cout << "\n(5) inherited Raytracer machinery (run_raytrace/redshift_start/redshift)" << endl;
    {
        LogImagePlane<double> small(dist, incl, 2.0, 30.0, 4, 2, 2.0, 30.0, 4, 2, spin, phi0);
        small.redshift_start();
        small.run_raytrace(Integrator::Euler, M_PI_2, 1.1 * dist, 0);
        small.redshift(true);
        int nsteps_pos = 0;
        for (int ix = 0; ix < small.get_count(); ix++)
            if (small.rays[ix].steps > 0) ++nsteps_pos;
        cout << "  " << small.get_count() << " rays, " << nsteps_pos << " with steps > 0" << endl;
        check(small.get_count() == small.get_Nx() * small.get_Ny(), "(5) ray count matches Nx_total*Ny_total");
        check(nsteps_pos > small.get_count() / 2, "(5) run_raytrace/redshift_start/redshift run without crashing and produce sensible steps");
    }

    cout << endl;
    if (failures == 0) cout << "ALL CHECKS PASSED (0 failures)" << endl;
    else cout << failures << " CHECK(S) FAILED" << endl;
    return failures == 0 ? 0 : 1;
}
