//
// ray_transfer_grid_test.cpp
//
// Self-checking tests for ImagePlane's/LogImagePlane's virtual pixel-grid query interface
// (get_Nx/get_Ny/pixel_x/pixel_y/pixel_dx/pixel_dy/pixel_weight, imageplane.h/log_imageplane.h) -- the
// mechanism RayTransfer uses to drive either grid type through a single const ImagePlane<T>& reference,
// with no separate adapter object:
//
//   (1) A plain ImagePlane's pixel_x/pixel_y/pixel_dx/pixel_dy reproduce the exact historical pixel-center
//       formula; pixel_weight() == 1 everywhere (the mechanism that keeps every existing linear-grid run
//       bit-identical).
//   (2) A LogImagePlane's pixel_x/pixel_y/pixel_weight -- called through a const ImagePlane<double>&
//       reference, to confirm they are genuinely virtual dispatch, not just correct when called directly --
//       match its own ray_x/ray_y/ray_weight at the equivalent flat ray index, and
//       pixel_dx(ix)*pixel_dy(iy) == pixel_weight(ix,iy).
//   (3) End-to-end: RayTransfer built from a LogImagePlane matches an independent re-derivation of
//       continuum_total/spec_total computed by calling plane.pixel_x/pixel_y/pixel_weight directly (no
//       separate grid object at all) -- direct proof of what run_raytrace() actually applies.
//   (4) Backward-compat spot check: RayTransfer built from a plain ImagePlane matches an unweighted
//       hand-rolled loop over trace_pixel(), calling plane.pixel_x/pixel_y directly.
//   (5) Regression test for a real bug found while designing this: since ImagePlane::init_ray is now
//       virtual, RayTransfer::trace_pixel's m_plane.init_ray(...) call correctly reaches
//       LogImagePlane::init_ray's origin guard through a const ImagePlane<T>& reference (the shape
//       RayTransfer actually stores it in) -- before init_ray was virtual, that call always silently
//       resolved to ImagePlane::init_ray's own unguarded (and, at x=y=0, NaN-producing) formula instead.
//
// Build (from the project root, no cfitsio needed):
//   g++ -O2 -std=c++17 -fopenmp -I src src/tests/ray_transfer_grid_test.cpp \
//       src/ray_transfer/ray_transfer.cpp src/raytracer/raytracer.cpp src/raytracer/imageplane.cpp \
//       src/raytracer/log_imageplane.cpp -o bin/ray_transfer_grid_test
//
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

#include "ray_transfer/ray_transfer.h"
#include "raytracer/log_imageplane.h"

static int failures = 0;
static void check(bool ok, const string& msg)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << msg << endl;
    if (!ok) ++failures;
}

int main()
{
    const double dist = 1000, incl = 60, spin_lin = 0.9, phi0 = 0;

    // (1) ImagePlane's own pixel-grid query
    cout << "(1) ImagePlane pixel-grid query" << endl;
    {
        const double x0 = -10.0, xmax = 10.0, dx = 2.0, y0 = -8.0, ymax = 8.0, dy = 4.0;
        ImagePlane<double> plane(dist, incl, x0, xmax, dx, y0, ymax, dy, spin_lin, phi0);
        bool ok = true;
        for (int ix = 0; ix < plane.get_Nx(); ix++)
            for (int iy = 0; iy < plane.get_Ny(); iy++)
            {
                if (plane.pixel_weight(ix, iy) != 1.0) ok = false;
                if (plane.pixel_x(ix) != x0 + (ix + 0.5) * dx) ok = false;
                if (plane.pixel_y(iy) != y0 + (iy + 0.5) * dy) ok = false;
                if (plane.pixel_dx(ix) != dx || plane.pixel_dy(iy) != dy) ok = false;
            }
        check(ok, "pixel_x/pixel_y/pixel_dx/pixel_dy reproduce the exact pixel-center formula; pixel_weight() == 1 everywhere");
    }

    // Regression test for a real bug found while building this: get_Nx()/get_Ny() must NOT return
    // ImagePlane's own private Nx/Ny (computed via its "grid point" formula, Nx = ((xmax-x0)/dx)+1 -- one
    // MORE than the pixel-center count) -- doing so silently traced an extra, out-of-range row/column and
    // shifted every aggregate sum by a few percent (caught by the git-stash bit-identical check against
    // ray_transfer_kerr_vs_flat's pre-existing output). Uses dx values that are not exact binary fractions
    // (mirroring ray_transfer_kerr_vs_flat.par_example's x0 = -1.2*R_out, xmax = 1.2*R_out, Nx = 60) to
    // stress-test the round-trip dx = (xmax-x0)/N -> get_Nx() == N.
    cout << "\n(1b) get_Nx()/get_Ny() recover the exact intended pixel count (not ImagePlane's own +1 grid-point Nx)" << endl;
    {
        bool ok = true;
        for (int N : {1, 3, 7, 17, 60, 101})
        {
            const double x0 = -1.2 * 37.0, xmax = 1.2 * 37.0;   // R_out = 37, an arbitrary non-round value
            const double dx = (xmax - x0) / N;
            ImagePlane<double> plane(dist, incl, x0, xmax, dx, x0, xmax, dx, spin_lin, phi0);
            if (plane.get_Nx() != N || plane.get_Ny() != N) ok = false;
        }
        check(ok, "get_Nx()/get_Ny() == N for dx = (xmax-x0)/N, across several N");
    }

    // (2) LogImagePlane's overrides, reached through a base reference
    cout << "\n(2) LogImagePlane pixel-grid query, via a const ImagePlane<double>& reference" << endl;
    {
        LogImagePlane<double> plane(dist, incl, 2.0, 100.0, 15, 4, 3.0, 80.0, 12, 6, spin_lin, phi0);
        const ImagePlane<double>& base = plane;   // the shape RayTransfer actually stores m_plane as
        bool ok = true;
        for (int ix = 0; ix < base.get_Nx(); ix++)
            for (int iy = 0; iy < base.get_Ny(); iy++)
            {
                const int flat = ix * plane.get_Ny() + iy;
                if (base.pixel_x(ix) != plane.ray_x(flat)) ok = false;
                if (base.pixel_y(iy) != plane.ray_y(flat)) ok = false;
                if (base.pixel_weight(ix, iy) != plane.ray_weight(flat)) ok = false;
                const double w = base.pixel_weight(ix, iy);
                if (fabs(base.pixel_dx(ix) * base.pixel_dy(iy) - w) > 1e-9 * w) ok = false;
            }
        check(base.get_Nx() == plane.get_Nx() && base.get_Ny() == plane.get_Ny(),
              "get_Nx()/get_Ny() through the base reference match LogImagePlane's own");
        check(ok, "pixel_x/pixel_y/pixel_weight through the base reference match ray_x/ray_y/ray_weight exactly "
                  "(genuine virtual dispatch, not ImagePlane's own formula); pixel_dx*pixel_dy == pixel_weight");
    }

    // Shared setup for (3) and (4): a Schwarzschild corona + a real (nonzero) spherical wind, so
    // spec_line/spec_total have genuine, non-trivial content to check the weighting against.
    const double spin = 0.0;
    SphericalContinuumSource<double> corona(5.0, 1.0);
    SphericalBetaWind<double> wind(0.05, 0.05 * 0.05, 1.0, 6.0, 100.0, 1.0);
    LineTransition<double> line{1.0, 0.02, 0.2};
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(0.8, 1.2, 5);

    // (4) Backward-compat spot check -- RayTransfer's single constructor, given a plain ImagePlane
    cout << "\n(4) RayTransfer(ImagePlane) vs an unweighted hand-rolled sum" << endl;
    {
        ImagePlane<double> plane(dist, incl, -8, 8, 2.0, -8, 8, 2.0, spin, phi0);
        RayTransfer<double> rt(plane, spin, wind, line, bins, nullptr, &corona);
        rt.run_raytrace(-1, -1, 0);
        check(rt.get_Nx() == plane.get_Nx() && rt.get_Ny() == plane.get_Ny(), "rt.get_Nx()/get_Ny() match plane's");

        const size_t ne = bins.energy.size();
        vector<double> ref_total(ne, 0.0);
        double ref_continuum_total = 0;
        vector<double> em, ab;
        for (int ix = 0; ix < plane.get_Nx(); ix++)
            for (int iy = 0; iy < plane.get_Ny(); iy++)
            {
                double continuum;
                rt.trace_pixel(plane.pixel_x(ix), plane.pixel_y(iy), em, ab, continuum);
                ref_continuum_total += continuum;
                for (size_t j = 0; j < ne; j++)
                    ref_total[j] += continuum * exp(-ab[j]) + em[j];
            }

        double max_dev = fabs(ref_continuum_total - rt.continuum_total);
        for (size_t j = 0; j < ne; j++)
            max_dev = max(max_dev, fabs(ref_total[j] - rt.spec_total[j]));
        // Not bit-exact: run_raytrace()'s per-row partial sums merged under a critical section accumulate
        // in a different order than this flat, serial ix/iy loop (floating-point addition isn't
        // associative), so a few ULPs of difference is expected even with weight == 1 throughout -- the
        // check that matters is that the *weighting* introduces no discrepancy beyond that summation-order
        // noise, i.e. it really is an unweighted sum, not that it's bit-identical to one particular
        // summation order.
        const double scale = max(1.0, fabs(rt.continuum_total));
        cout << "  max |diff| = " << scientific << max_dev << fixed << endl;
        check(max_dev < 1e-9 * scale, "run_raytrace()'s continuum_total/spec_total match an unweighted hand-rolled sum");
    }

    // (3) End-to-end log-grid weighting proof -- RayTransfer's single constructor, given a LogImagePlane
    cout << "\n(3) RayTransfer(LogImagePlane) vs an independent weighted re-derivation" << endl;
    {
        LogImagePlane<double> plane(dist, incl, 2.0, 20.0, 10, 4, 2.0, 20.0, 10, 4, spin, phi0);
        RayTransfer<double> rt(plane, spin, wind, line, bins, nullptr, &corona);
        rt.run_raytrace(-1, -1, 0);
        check(rt.get_Nx() == plane.get_Nx() && rt.get_Ny() == plane.get_Ny(), "rt.get_Nx()/get_Ny() match plane's");

        const size_t ne = bins.energy.size();
        double re_continuum_total = 0;
        vector<double> re_total(ne, 0.0);
        double wmin = 1e300, wmax = -1e300;
        for (int ix = 0; ix < plane.get_Nx(); ix++)
            for (int iy = 0; iy < plane.get_Ny(); iy++)
            {
                const double w = plane.pixel_weight(ix, iy);
                wmin = min(wmin, w); wmax = max(wmax, w);
                re_continuum_total += (*rt.continuum_map)[ix][iy] * w;
                for (size_t j = 0; j < ne; j++)
                    re_total[j] += (*rt.flux_cube)[j][iy][ix] * w;
            }

        double max_dev = fabs(re_continuum_total - rt.continuum_total);
        for (size_t j = 0; j < ne; j++)
            max_dev = max(max_dev, fabs(re_total[j] - rt.spec_total[j]));
        const double scale = max(1.0, fabs(rt.continuum_total));
        cout << "  max |diff| = " << scientific << max_dev << fixed
             << ", pixel weight range [" << wmin << ", " << wmax << "]" << endl;
        check(max_dev < 1e-9 * scale, "run_raytrace()'s continuum_total/spec_total match plane.pixel_weight()-based re-derivation");
        check(wmax > wmin * 1.5, "log grid pixel weights are genuinely non-uniform (a real test of the weighting, not a near-linear case)");
    }

    // (5) Regression test: init_ray must be virtual for LogImagePlane's origin guard to be reachable
    // through a const ImagePlane<T>& reference -- the shape RayTransfer actually holds m_plane in.
    cout << "\n(5) init_ray virtual dispatch: LogImagePlane's origin guard reached through a base reference" << endl;
    {
        // Nlinx=Nliny=1: the single linear-zone bin's center is exactly 0 in floating point for any lo
        // (multiplying/dividing by 2 is exact in IEEE754) on both axes, guaranteeing a ray exactly at x=y=0.
        LogImagePlane<double> plane(dist, incl, 2.0, 20.0, 5, 1, 2.0, 20.0, 5, 1, spin, phi0);
        bool hit_origin = false;
        for (int ix = 0; ix < plane.get_Nx() && !hit_origin; ix++)
            for (int iy = 0; iy < plane.get_Ny() && !hit_origin; iy++)
                if (plane.pixel_x(ix) == 0.0 && plane.pixel_y(iy) == 0.0) hit_origin = true;
        check(hit_origin, "test grid genuinely places a pixel exactly at x=y=0");

        // Direct check: init_ray dispatch through a base reference.
        const ImagePlane<double>& base = plane;
        Ray<double> ray;
        base.init_ray(ray, 0.0, 0.0, dist, incl * M_PI / 180, phi0);
        check(std::isfinite(ray.h) && std::isfinite(ray.Q),
              "ImagePlane&::init_ray(0,0,...) is finite -- reaches LogImagePlane's origin guard via virtual dispatch");

        // End-to-end: RayTransfer::trace_pixel at that exact (x,y) should not be silently dropped either.
        RayTransfer<double> rt(plane, spin, wind, line, bins, nullptr, &corona);
        vector<double> em, ab;
        double continuum;
        rt.trace_pixel(0.0, 0.0, em, ab, continuum);
        bool all_finite = std::isfinite(continuum);
        for (double v : em) if (!std::isfinite(v)) all_finite = false;
        for (double v : ab) if (!std::isfinite(v)) all_finite = false;
        check(all_finite, "RayTransfer::trace_pixel(0, 0, ...) produces a finite result, not a silently-dropped NaN");
    }

    cout << endl << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << endl;
    return failures == 0 ? 0 : 1;
}
