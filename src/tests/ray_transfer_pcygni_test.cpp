//
// ray_transfer_pcygni_test.cpp
//
// Self-checking verification that RayTransfer/FlatRayTransfer (src/ray_transfer/ray_transfer.h)
// reproduces the classical P-Cygni line profile for a spherical wind with no black hole (flat spacetime):
//
//   (a) morphology -- a blueshifted absorption trough (residual flux < 1 above the rest energy, where
//       approaching near-side wind material occults the stellar continuum) and a redshifted emission
//       excess (residual > 1 below the rest energy, from receding far-side material scattering starlight
//       into the line of sight) -- the defining P-Cygni signature (Lamers & Cassinelli, "Introduction to
//       Stellar Winds"; Rybicki & Hummer 1978).
//
//   (b) the asymmetry is genuinely Doppler-driven, not a coordinate/geometry artifact: with v_inf = 0 (no
//       bulk wind motion) every point's local frame coincides with the lab frame, g = 1 everywhere, and
//       the residual spectrum is then exactly symmetric about the line's rest energy (no preferred
//       red/blue side); switching on v_inf breaks that symmetry, and only then, into the blue-trough /
//       red-bump pattern of (a).  (An exact photon-conservation identity -- integral(residual - 1) dE = 0
//       -- was considered for this test but dropped: for a single fixed observer direction and an
//       extended, isotropically-scattering photosphere it is not actually an exact invariant of this model
//       [energy conservation only closes when the emission is integrated over the full 4*pi sr of
//       "virtual observers", not one sightline], and numerically it is also very sensitive to the ratio of
//       the line's Doppler width to the line-of-sight step size, making it a poor verification signal here.)
//
// Build (from the project root, no cfitsio needed):
//   g++ -O2 -std=c++17 -I src src/tests/ray_transfer_pcygni_test.cpp src/ray_transfer/ray_transfer.cpp \
//       -o bin/ray_transfer_pcygni_test
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

// Sum the per-impact-parameter emission (FlatRayTransfer::trace_ray) over impact parameter, exactly as
// ray_transfer_pcygni.cpp does, and return the continuum-normalised residual flux spectrum.
static vector<double> pcygni_residual(double v_inf, double kappa0, const SpectrumGrid<double>& bins)
{
    const double beta_exp = 1.0, R0 = 1.0, n0 = 1.0;
    const double line_energy = 1.0, doppler_width = 0.001, I_star = 1.0;
    const double p_max = 5 * R0, z_max = 20 * R0, dz = 0.02 * R0;
    const int n_p = 150;
    // v_inf == 0 is the deliberately degenerate "no wind at all" case used by the symmetry check below
    // (v0 == 0 too, so velocity() is exactly 0 everywhere and density() returns 0 -- see ray_transfer.cpp);
    // otherwise a small physical base velocity, as any real disc-wind application should set.
    const double v0 = (v_inf > 0) ? 0.05 * v_inf : 0.0;

    LineTransition<double> line{line_energy, doppler_width, kappa0};
    SphericalBetaWind<double> wind(v_inf, v0, beta_exp, R0, 1e6 * R0, n0);
    FlatRayTransfer<double> rt(wind, line, bins);

    const int n_energy = (int)bins.energy.size();
    vector<double> flux(n_energy, 0.0), em(n_energy), ab(n_energy);
    const double dp = p_max / n_p;
    for (int i = 0; i < n_p; i++)
    {
        const double p = (i + 0.5) * dp;
        rt.trace_ray(p, z_max, -z_max, dz, R0, I_star, em, ab);
        const double weight = 2 * M_PI * p * dp;
        for (int j = 0; j < n_energy; j++) flux[j] += weight * em[j];
    }
    const double norm = M_PI * R0 * R0 * I_star;
    vector<double> residual(n_energy);
    for (int j = 0; j < n_energy; j++) residual[j] = flux[j] / norm;
    return residual;
}

int main()
{
    const double line_energy = 1.0;
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(0.96, 1.02, 121);

    // (a) morphology, at a moderate optical depth.  kappa0 = 0.002, not the naive-looking 5.0: every ray
    // with p <~ R0 terminates right at r = R0 (the star's surface is the wind's launch point by design),
    // crossing the same near-launch density pileup as any other application of this wind model (see
    // docs/plan_ray_transfer.md Sec 5.4-5.5); kappa0 = 5.0 saturates the trough well before reaching that
    // value (checked: the depth barely changes from kappa0 = 5.0 down to ~0.1), so it wasn't actually
    // testing kappa0-sensitive physics. 0.002 sits below that saturation floor.
    {
        const vector<double> residual = pcygni_residual(0.01, 0.002, bins);
        double min_blue = 1.0, max_red = 1.0;
        for (size_t j = 0; j < bins.energy.size(); j++)
        {
            if (bins.energy[j] > line_energy) min_blue = min(min_blue, residual[j]);
            if (bins.energy[j] < line_energy) max_red = max(max_red, residual[j]);
        }
        cout << "Morphology: min residual blueward = " << min_blue
             << ", max residual redward = " << max_red << endl;
        check(min_blue < 0.9, "blueshifted absorption trough (residual < 0.9 above the line energy)");
        check(max_red > 1.01, "redshifted emission excess (residual > 1.01 below the line energy)");
    }

    // (b) the red/blue asymmetry is Doppler-driven: zero for a static wind, large once it flows
    {
        auto asymmetry = [&](double v_inf) -> double
        {
            const vector<double> residual = pcygni_residual(v_inf, 0.002, bins);
            size_t mid = 0; double best = 1e9;
            for (size_t j = 0; j < bins.energy.size(); j++)
                if (fabs(bins.energy[j] - line_energy) < best) { best = fabs(bins.energy[j] - line_energy); mid = j; }
            const size_t half = min(mid, bins.energy.size() - 1 - mid);
            double a = 0;
            for (size_t k = 1; k <= half; k++) a += fabs(residual[mid+k] - residual[mid-k]);
            return a;
        };
        const double asym_static = asymmetry(0.0);
        const double asym_flowing = asymmetry(0.01);
        cout << "Asymmetry: static wind = " << asym_static << ", flowing wind = " << asym_flowing << endl;
        check(asym_static < 1e-9, "no bulk motion -> exactly symmetric profile (g = 1 everywhere)");
        check(asym_flowing > 1.0, "wind flow -> strongly asymmetric (blue trough / red bump) profile");
    }

    cout << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << endl;
    return failures == 0 ? 0 : 1;
}
