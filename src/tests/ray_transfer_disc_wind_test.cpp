//
// ray_transfer_disc_wind_test.cpp
//
// Consistency checks for RayTransfer<double>'s disc-blocking and corona-termination logic
// (src/ray_transfer/ray_transfer.h), added alongside the disc-wind application
// (src/ray_transfer/ray_transfer_disc_wind.cpp):
//
//   (a) Schwarzschild static-corona redshift -- an exact, independent formula.  At spin = 0, a static
//       observer's contravariant momentum component p^t = k / (1 - 2/r) for *any* photon regardless of its
//       angular momentum h (momentum_from_consts, kerr.h, collapses to this at a = 0), so the blueshift
//       of a photon falling from the image plane (effectively r = infinity) to a static corona at radius
//       r is g = E_loc/E_obs = 1/sqrt(1 - 2/r), independent of which pixel/direction the ray arrived from.
//       This checks Corona::four_velocity + the ray_redshift reuse in RayTransfer::trace_pixel against
//       that closed-form result, via continuum = I_corona / g^3 (the I_nu/nu^3 invariant specific-intensity
//       boost -- ray_transfer.cpp; g > 1 here, so the corona comes out dimmer than I_corona, as it must
//       for redshifted light).
//
//   (b) Disc blocking -- scanning a small image-plane grid at high inclination (disc close to edge-on),
//       at least some pixels must be blocked by the disc (continuum = 0, geometrically aimed through the
//       disc's radial range) and at least some must reach the corona (continuum > 0) -- i.e. the
//       DiscWithISCODestination reuse actually terminates rays, and doesn't block everything or nothing.
//
// Build (from the project root, no cfitsio needed):
//   g++ -O2 -std=c++17 -fopenmp -I src src/tests/ray_transfer_disc_wind_test.cpp \
//       src/ray_transfer/ray_transfer.cpp src/raytracer/raytracer.cpp src/raytracer/imageplane.cpp \
//       -o bin/ray_transfer_disc_wind_test
//
#include <iostream>
#include <vector>
#include <cmath>

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
    // (a) Schwarzschild static-corona redshift
    {
        const double spin = 0.0, R_corona = 5.0, I_corona = 1.0, dist = 1000.0;
        // E_loc(r) = k / sqrt(1 - 2/r) exactly, for a static observer at any r (a = 0); g is the ratio
        // between the corona and the image plane (at finite distance, not literally r = infinity)
        const double g_expected = sqrt(1 - 2.0/dist) / sqrt(1 - 2.0/R_corona);

        ImagePlane<double> plane(dist, 60.0, -6, 6, 1.0, -6, 6, 1.0, spin, 0.0);
        // v0/kappa0 are inert here (n0 = 0: wind switched off, to isolate the redshift formula from the
        // wind's density model), but kept at the same values used everywhere else the wind is actually
        // on (v0_vinf = 0.05, kappa0 = 0.2 -- see docs/plan_ray_transfer.md Sec 5.4-5.6) for consistency.
        SphericalBetaWind<double> wind(0.01, 0.05*0.01, 1.0, 6.0, 100.0, 0.0);
        SphericalCorona<double> corona(R_corona, I_corona);
        LineTransition<double> line{1.0, 0.01, 0.2};
        SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(0.9, 1.1, 3);
        RayTransfer<double> rt(plane, spin, wind, line, bins, nullptr, &corona);

        int n_hit = 0;
        double max_dev = 0;
        vector<double> em, ab;
        for (double x = -6; x <= 6; x += 1.0)
            for (double y = -6; y <= 6; y += 1.0)
            {
                double continuum;
                if (!rt.trace_pixel(x, y, em, ab, continuum)) continue;
                ++n_hit;
                // continuum = I_corona / g^3 (I_nu/nu^3 invariance -- ray_transfer.cpp), so g = (I_corona/continuum)^(1/3)
                const double g = pow(I_corona / continuum, 1.0/3.0);
                max_dev = max(max_dev, fabs(g - g_expected) / g_expected);
            }
        cout << "Schwarzschild corona: " << n_hit << " pixels hit the corona, g_expected = " << g_expected
             << ", max relative deviation = " << max_dev << endl;
        check(n_hit > 0, "at least some pixels reach the corona");
        check(max_dev < 1e-6, "static-observer blueshift matches 1/sqrt(1 - 2/R_corona) to 1e-6");
    }

    // (b) disc blocking: some pixels blocked, some reach the corona
    {
        const double spin = 0.9;
        const double r_isco = kerr_isco<double>(spin, +1);
        const double R_corona = 0.5 * (kerr_horizon<double>(spin) + r_isco);   // strictly between horizon and ISCO
        const double r_out_disc = 30.0;

        ImagePlane<double> plane(1000.0, 85.0, -40, 40, 4.0, -15, 15, 3.0, spin, 0.0);
        // v0/kappa0 are inert here too (n0 = 0: isolate disc blocking from the wind's density model), same
        // consistency note as above.
        SphericalBetaWind<double> wind(0.01, 0.05*0.01, 1.0, r_isco, 100.0, 0.0);
        SphericalCorona<double> corona(R_corona, 1.0);
        DiscWithISCODestination<double> disc(r_isco, r_out_disc);
        LineTransition<double> line{1.0, 0.01, 0.2};
        SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(0.9, 1.1, 3);
        RayTransfer<double> rt(plane, spin, wind, line, bins, &disc, &corona);

        int n_hit = 0, n_blocked = 0, n_other = 0;
        vector<double> em, ab;
        for (double x = -40; x <= 40; x += 4.0)
            for (double y = -15; y <= 15; y += 3.0)
            {
                double continuum;
                const bool hit = rt.trace_pixel(x, y, em, ab, continuum);
                if (hit) ++n_hit;
                else if (fabs(y) < 3.0) ++n_blocked;   // near the midplane at high inclination: expect the disc
                else ++n_other;
            }
        cout << "Disc/corona scan: " << n_hit << " hit corona, " << n_blocked << " blocked near midplane, "
             << n_other << " other (escaped/step-limited)" << endl;
        check(n_hit > 0, "at least some pixels reach the corona (not blocked by everything)");
        check(n_blocked > 0, "at least some near-midplane pixels are blocked by the disc");
    }

    cout << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << endl;
    return failures == 0 ? 0 : 1;
}
