//
// ray_transfer_grid_test.cpp
//
// Self-checking tests for RayTransferGrid/LinearRayTransferGrid/LogRayTransferGrid (ray_transfer.h) --
// added alongside making LogImagePlane an option in RayTransfer:
//
//   (1) LinearRayTransferGrid reproduces RayTransfer's original pixel-center formula exactly, and
//       weight() == 1 always (the mechanism that keeps every existing linear-grid run bit-identical).
//   (2) LogRayTransferGrid's x/y/weight exactly match LogImagePlane::ray_x/ray_y/ray_weight at the
//       equivalent flat ray index, and dx(ix)*dy(iy) == weight(ix,iy) for the log grid.
//   (3) End-to-end: RayTransfer::run_raytrace(), constructed via its LogImagePlane-only overload, produces
//       continuum_total/spec_total that exactly match an independent re-derivation from continuum_map/
//       flux_cube weighted by a standalone LogRayTransferGrid's weight(ix,iy) -- direct proof of what
//       run_raytrace() actually applies, and that RayTransfer::grid_x/grid_y/grid_dx/grid_dy/grid_weight
//       (the public pass-throughs to the grid it built internally) agree with that standalone grid exactly.
//   (4) Backward-compat spot check: RayTransfer::run_raytrace(), constructed via its original ImagePlane +
//       Nx/Ny/x0/dx/y0/dy overload, matches an unweighted hand-rolled loop over trace_pixel() exactly --
//       proving the linear path's aggregate numbers (and constructor signature) are unchanged from before
//       grid_type existed.
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

static int failures = 0;
static void check(bool ok, const string& msg)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << msg << endl;
    if (!ok) ++failures;
}

int main()
{
    // (1) LinearRayTransferGrid
    cout << "(1) LinearRayTransferGrid" << endl;
    {
        LinearRayTransferGrid<double> grid(5, 4, -10.0, 2.0, -8.0, 4.0);
        bool ok = true;
        for (int ix = 0; ix < grid.Nx(); ix++)
            for (int iy = 0; iy < grid.Ny(); iy++)
            {
                if (grid.weight(ix, iy) != 1.0) ok = false;
                if (grid.x(ix) != -10.0 + (ix + 0.5) * 2.0) ok = false;
                if (grid.y(iy) != -8.0 + (iy + 0.5) * 4.0) ok = false;
                if (grid.dx(ix) != 2.0 || grid.dy(iy) != 4.0) ok = false;
            }
        check(grid.Nx() == 5 && grid.Ny() == 4, "Nx()/Ny() report the constructor's values");
        check(ok, "x/y/dx/dy reproduce the exact pixel-center formula; weight() == 1 everywhere");
    }

    // (2) LogRayTransferGrid vs LogImagePlane
    cout << "\n(2) LogRayTransferGrid vs LogImagePlane" << endl;
    {
        const double dist = 1000, incl = 60, spin = 0.9, phi0 = 0;
        LogImagePlane<double> plane(dist, incl, 2.0, 100.0, 15, 4, 3.0, 80.0, 12, 6, spin, phi0);
        LogRayTransferGrid<double> grid(plane);
        bool ok = true;
        for (int ix = 0; ix < grid.Nx(); ix++)
            for (int iy = 0; iy < grid.Ny(); iy++)
            {
                const int flat = ix * plane.get_Ny() + iy;
                if (grid.x(ix) != plane.ray_x(flat)) ok = false;
                if (grid.y(iy) != plane.ray_y(flat)) ok = false;
                if (grid.weight(ix, iy) != plane.ray_weight(flat)) ok = false;
                const double w = grid.weight(ix, iy);
                if (fabs(grid.dx(ix) * grid.dy(iy) - w) > 1e-9 * w) ok = false;
            }
        check(grid.Nx() == plane.get_Nx() && grid.Ny() == plane.get_Ny(), "Nx()/Ny() match LogImagePlane's");
        check(ok, "x/y/weight match LogImagePlane::ray_x/ray_y/ray_weight exactly; dx(ix)*dy(iy) == weight(ix,iy)");
    }

    // Shared setup for (3) and (4): a Schwarzschild corona + a real (nonzero) spherical wind, so
    // spec_line/spec_total have genuine, non-trivial content to check the weighting against.
    const double dist = 1000, incl = 60, spin = 0.0, phi0 = 0;
    SphericalContinuumSource<double> corona(5.0, 1.0);
    SphericalBetaWind<double> wind(0.05, 0.05 * 0.05, 1.0, 6.0, 100.0, 1.0);
    LineTransition<double> line{1.0, 0.02, 0.2};
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(0.8, 1.2, 5);

    // (4) Backward-compat spot check -- RayTransfer's original constructor overload (ImagePlane +
    // Nx/Ny/x0/dx/y0/dy), unchanged since before grid_type existed.
    cout << "\n(4) ImagePlane-overload run_raytrace() vs an unweighted hand-rolled sum" << endl;
    {
        ImagePlane<double> plane(dist, incl, -8, 8, 2.0, -8, 8, 2.0, spin, phi0);
        RayTransfer<double> rt(plane, spin, wind, line, bins, 8, 8, -8.0, 2.0, -8.0, 2.0, nullptr, &corona);
        rt.run_raytrace(-1, -1, 0);

        // Independent reference grid -- not passed to RayTransfer at all, just used to drive this test's own
        // hand-rolled loop and to cross-check rt's grid_x/grid_y/grid_dx/grid_dy pass-throughs below.
        LinearRayTransferGrid<double> grid(8, 8, -8.0, 2.0, -8.0, 2.0);

        const size_t ne = bins.energy.size();
        vector<double> ref_total(ne, 0.0);
        double ref_continuum_total = 0;
        vector<double> em, ab;
        bool grid_accessors_ok = true;
        for (int ix = 0; ix < grid.Nx(); ix++)
            for (int iy = 0; iy < grid.Ny(); iy++)
            {
                if (rt.grid_x(ix) != grid.x(ix) || rt.grid_y(iy) != grid.y(iy)
                    || rt.grid_dx(ix) != grid.dx(ix) || rt.grid_dy(iy) != grid.dy(iy)
                    || rt.grid_weight(ix, iy) != grid.weight(ix, iy))
                    grid_accessors_ok = false;
                double continuum;
                rt.trace_pixel(grid.x(ix), grid.y(iy), em, ab, continuum);
                ref_continuum_total += continuum;
                for (size_t j = 0; j < ne; j++)
                    ref_total[j] += continuum * exp(-ab[j]) + em[j];
            }
        check(rt.get_Nx() == grid.Nx() && rt.get_Ny() == grid.Ny() && grid_accessors_ok,
              "get_Nx/get_Ny/grid_x/grid_y/grid_dx/grid_dy/grid_weight match the grid this constructor built internally");

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

    // (3) End-to-end log-grid weighting proof -- RayTransfer's LogImagePlane-only overload.
    cout << "\n(3) LogImagePlane-overload run_raytrace() vs an independent weighted re-derivation" << endl;
    {
        LogImagePlane<double> plane(dist, incl, 2.0, 20.0, 10, 4, 2.0, 20.0, 10, 4, spin, phi0);
        RayTransfer<double> rt(plane, spin, wind, line, bins, nullptr, &corona);
        rt.run_raytrace(-1, -1, 0);

        // Independent reference grid -- not passed to RayTransfer (this overload takes none), just used to
        // drive this test's own weighted re-derivation and to cross-check rt's pass-throughs below.
        LogRayTransferGrid<double> grid(plane);
        check(rt.get_Nx() == grid.Nx() && rt.get_Ny() == grid.Ny(), "get_Nx/get_Ny match the grid this constructor built internally");

        const size_t ne = bins.energy.size();
        double re_continuum_total = 0;
        vector<double> re_total(ne, 0.0);
        double wmin = 1e300, wmax = -1e300;
        bool grid_accessors_ok = true;
        for (int ix = 0; ix < grid.Nx(); ix++)
            for (int iy = 0; iy < grid.Ny(); iy++)
            {
                if (rt.grid_x(ix) != grid.x(ix) || rt.grid_y(iy) != grid.y(iy)
                    || rt.grid_dx(ix) != grid.dx(ix) || rt.grid_dy(iy) != grid.dy(iy)
                    || rt.grid_weight(ix, iy) != grid.weight(ix, iy))
                    grid_accessors_ok = false;
                const double w = grid.weight(ix, iy);
                wmin = min(wmin, w); wmax = max(wmax, w);
                re_continuum_total += (*rt.continuum_map)[ix][iy] * w;
                for (size_t j = 0; j < ne; j++)
                    re_total[j] += (*rt.flux_cube)[j][iy][ix] * w;
            }
        check(grid_accessors_ok, "grid_x/grid_y/grid_dx/grid_dy/grid_weight match the grid this constructor built internally");

        double max_dev = fabs(re_continuum_total - rt.continuum_total);
        for (size_t j = 0; j < ne; j++)
            max_dev = max(max_dev, fabs(re_total[j] - rt.spec_total[j]));
        const double scale = max(1.0, fabs(rt.continuum_total));
        cout << "  max |diff| = " << scientific << max_dev << fixed
             << ", pixel weight range [" << wmin << ", " << wmax << "]" << endl;
        check(max_dev < 1e-9 * scale, "run_raytrace()'s continuum_total/spec_total match grid.weight()-based re-derivation");
        check(wmax > wmin * 1.5, "log grid pixel weights are genuinely non-uniform (a real test of the weighting, not a near-linear case)");
    }

    cout << endl << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << endl;
    return failures == 0 ? 0 : 1;
}
