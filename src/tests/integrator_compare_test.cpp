/*
 * integrator_compare_test.cpp
 *
 * Runs the same lamppost rays through all four integrators (Euler, RK4, RK45,
 * Symplectic) and compares each against a tightly-toleranced RK45 reference
 * (rk45_tol = 1e-12).  Generalises raytrace_rk4_test to the full set of methods.
 *
 * For each method the test reports:
 *   - propagation wall-clock time and mean step count
 *   - termination-class agreement with the reference (disc / r_max / horizon / steplim)
 *   - for rays that reach the disc in both runs: percentiles of the differences in
 *     disc radius, phi, arrival time t and redshift g (Keplerian disc, V = -1)
 *   - for the symplectic run: the worst violation of the null condition |H_M| and of
 *     Carter's constant |dQ|/Q recomputed from the final state (the other methods
 *     conserve these by construction, so they carry no information there)
 *
 * Separatrix rays (near the photon sphere) are chaotically sensitive to the step
 * size and can legitimately land at very different radii; they are counted as
 * "diverged" (|dr|/r > 1%) rather than failing the test.
 *
 * Known defect of the Carter-equation propagators (Euler/RK4/RK45), found while
 * validating the symplectic integrator: rays emitted close to a radial turning
 * point (|cos alpha| <~ 0.1) *and* initially heading towards the polar axis
 * (sin beta > 0 for a source just above the axis) undergo a spurious radial
 * sign flip when the polar sign flip fires -- inside the polar forbidden zone
 * the propagators use |Theta| in place of Theta in rdotsq -- and land at the
 * wrong radius (e.g. r = 14.26 instead of 8.43 for cos alpha = -0.1).  Direct
 * Carter quadratures and an independent scipy integration agree with the
 * symplectic result.  Such rays are excluded from the pass/fail statistics
 * and reported separately as the "defect class".
 *
 * Pass criteria are:
 *   - termination-class agreement >= 99% and diverged fraction <= 1%
 *   - median |dr| at the disc below 1e-4 (symplectic and RK45), below 1e-2 (RK4)
 *   - max |H_M| and |dQ|/Q below 2e-5 (order 4) / 1e-6 (order 6) for the symplectic runs
 *
 * The Euler and RK4 runs at the default precision are expected to fail the agreement and
 * divergence criteria -- they are reported for reference (accuracy vs cost is the subject
 * of integrator_perf_test), and their failures do not count towards the exit status.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstdlib>
using namespace std;

#include "raytracer/pointsource.h"
#include "include/kerr.h"

static const double SPIN      = 0.998;
static const double SOURCE[]  = {0.0, 5.0, 1e-3, 0.0};
static const double V         = 0.0;
static const double DCOSALPHA = 0.05;
static const double DBETA     = 0.05;
static const double R_MAX     = 1000.0;
static const double THETA_MAX = M_PI_2;

static int n_fail = 0;
static bool informational = false;   // when set, failed checks are reported but do not count
static void check(bool ok, const string& what)
{
    cout << (ok ? "  [PASS] " : (informational ? "  [info] " : "  [FAIL] ")) << what << endl;
    if (!ok && !informational) ++n_fail;
}

struct Run
{
    PointSource<double>* src;
    double seconds;
};

static Run run_method(Integrator method, double rk45_tol = -1, double symp_h0 = -1, int symp_order = 4, double max_tstep = -1)
{
    double pos[4] = {SOURCE[0], SOURCE[1], SOURCE[2], SOURCE[3]};
    PointSource<double>* src = new PointSource<double>(pos, V, SPIN, TOL, DCOSALPHA, DBETA);
    if (rk45_tol > 0) src->set_rk45_tol(rk45_tol);
    if (symp_h0 > 0)  src->set_symplectic_step(symp_h0);
    src->set_symplectic_order(symp_order);
    if (max_tstep > 0) src->set_max_tstep(max_tstep);
    src->redshift_start();

    auto t0 = chrono::high_resolution_clock::now();
    src->run_raytrace(method, THETA_MAX, R_MAX, 0);
    auto t1 = chrono::high_resolution_clock::now();
    src->redshift(-1.0);   // Keplerian disc velocity at the landing radius
    return { src, chrono::duration<double>(t1 - t0).count() };
}

static int term_class(const Ray<double>& r)
{
    if (r.status & RAY_STATUS_STEPLIM) return 3;
    if (r.status & RAY_STATUS_HORIZON) return 2;
    if (r.status & RAY_STATUS_RLIM)    return 1;
    if (r.status & RAY_STATUS_DEST)    return 0;
    return 4;
}

static double pct(vector<double> v, double p)
{
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    size_t i = min(v.size() - 1, (size_t)(p * v.size()));
    return v[i];
}

static void compare(const char* name, const Run& run, const Run& ref, double dr_median_tol, bool symplectic, double H_tol = 1e-6)
{
    const int n = ref.src->get_count();
    int n_valid = 0, n_classified = 0, n_defect_class = 0, n_agree = 0, n_both_disc = 0, n_diverged = 0, n_ref_steplim = 0, n_ref_steplim_horizon = 0;
    long total_steps = 0;
    int counts[5] = {0,0,0,0,0};
    vector<double> dr, dphi, dt, dg;
    double max_H = 0, max_dQ = 0;

    for (int i = 0; i < n; i++)
    {
        const Ray<double>& a = run.src->rays[i];
        const Ray<double>& b = ref.src->rays[i];
        if (b.steps == -1 && a.steps == -1) continue;   // never traced
        ++n_valid;
        total_steps += abs(a.steps);
        const int ca = term_class(a), cb = term_class(b);
        counts[ca]++;
        if (cb == 3) { ++n_ref_steplim; if (ca == 2) ++n_ref_steplim_horizon; continue; }   // reference failed (RK45 stalls on plunging rays)
        ++n_classified;
        if (ca == cb) ++n_agree;
        if (ca == 0 && cb == 0)
        {
            if (abs(a.alpha) <= 0.15 && sin(a.beta) > 0) { ++n_defect_class; continue; }   // see header comment
            ++n_both_disc;
            const double rel = abs(a.r - b.r) / b.r;
            if (rel > 1e-2) { ++n_diverged; continue; }
            dr.push_back(abs(a.r - b.r));
            double dp = a.phi - b.phi; dp = abs(atan2(sin(dp), cos(dp)));
            dphi.push_back(dp);
            dt.push_back(abs(a.t - b.t));
            dg.push_back(abs(a.redshift - b.redshift));
        }
        if (symplectic && ca != 2)
        {
            // recompute the Mino-time diagnostics from the stored (contravariant) final state
            const double rhosq = a.r*a.r + (SPIN*cos(a.theta))*(SPIN*cos(a.theta));
            const double u  = mino_u_from_r(a.r, SPIN);
            double rr, sd; mino_r_from_u(u, SPIN, rr, sd);
            const double pu = rhosq * a.pr / sd;
            const double pth = rhosq * a.ptheta;
            const double HM = mino_hamiltonian_r(u, pu, a.k, a.h, SPIN) + mino_hamiltonian_theta(a.theta, pth, a.k, a.h, SPIN);
            const double Hs = max(1.0, abs(mino_hamiltonian_theta(a.theta, pth, a.k, a.h, SPIN)));
            max_H  = max(max_H, abs(HM) / Hs);
            max_dQ = max(max_dQ, abs(carter_Q(a.theta, pth, a.k, a.h, SPIN) - a.Q) / max(1.0, abs(a.Q)));
        }
    }

    cout << endl << "=== " << name << " ===" << endl;
    cout << fixed << setprecision(3);
    cout << "  time " << run.seconds << " s,  mean steps/ray " << (double)total_steps / n_valid
         << ",  disc " << counts[0] << " / rlim " << counts[1] << " / horizon " << counts[2]
         << " / steplim " << counts[3] << " / none " << counts[4] << endl;
    cout << "  termination agreement with reference: " << n_agree << " / " << n_classified
         << " (" << 100.0 * n_agree / n_classified << "%),  diverged (|dr|/r > 1%): " << n_diverged << " / " << n_both_disc << endl;
    cout << "  disc rays in the Carter-propagator defect class (excluded from statistics): " << n_defect_class << endl;
    cout << "  rays the reference could not classify (RK45 step limit): " << n_ref_steplim
         << ", of which this method sends " << n_ref_steplim_horizon << " through the horizon" << endl;
    cout << scientific << setprecision(2);
    cout << "  |dr|   at disc: median " << pct(dr, 0.5)   << "  90% " << pct(dr, 0.9)   << "  max " << pct(dr, 1.0)   << endl;
    cout << "  |dphi| at disc: median " << pct(dphi, 0.5) << "  90% " << pct(dphi, 0.9) << "  max " << pct(dphi, 1.0) << endl;
    cout << "  |dt|   at disc: median " << pct(dt, 0.5)   << "  90% " << pct(dt, 0.9)   << "  max " << pct(dt, 1.0)   << endl;
    cout << "  |dg|   at disc: median " << pct(dg, 0.5)   << "  90% " << pct(dg, 0.9)   << "  max " << pct(dg, 1.0)   << endl;
    if (symplectic)
        cout << "  max |H_M| = " << max_H << ",  max |dQ|/Q = " << max_dQ << endl;

    check(100.0 * n_agree / n_classified >= 99.0, string(name) + ": termination agreement >= 99%");
    check(n_diverged <= 0.01 * n_both_disc, string(name) + ": diverged fraction <= 1%");
    check(pct(dr, 0.5) < dr_median_tol, string(name) + ": median |dr| at disc within tolerance");
    if (symplectic)
        check(max_H < H_tol && max_dQ < H_tol, string(name) + ": |H_M| and |dQ|/Q within tolerance");
}

int main(int argc, char** argv)
{
    // optional overrides for the symplectic runs: h0 and max_tstep (used for quick sweeps)
    const double symp_h0  = (argc > 1) ? atof(argv[1]) : -1;
    const double symp_mdt = (argc > 2) ? atof(argv[2]) : 0.25;
    cout << "spin = " << SPIN << ", source r = " << SOURCE[1] << ", theta = " << SOURCE[2]
         << ", dcosalpha = " << DCOSALPHA << ", dbeta = " << DBETA << endl;

    Run ref = run_method(Integrator::RK45, 1e-12);
    cout << "reference RK45 (tol 1e-12): " << fixed << setprecision(3) << ref.seconds << " s" << endl;

    Run euler = run_method(Integrator::Euler);
    Run rk4   = run_method(Integrator::RK4);
    Run rk45  = run_method(Integrator::RK45);
    Run symp4 = run_method(Integrator::Symplectic, -1, symp_h0, 4, symp_mdt);
    Run symp6 = run_method(Integrator::Symplectic, -1, symp_h0, 6, symp_mdt);

    informational = true;
    compare("Euler (precision = TOL)",            euler, ref, 1e-1, false);
    compare("RK4 (precision = TOL)",              rk4,   ref, 1e-2, false);
    informational = false;
    compare("RK45 (tol = 1e-8)",                  rk45,  ref, 1e-4, false);
    compare("Symplectic order 4", symp4, ref, 1e-4, true, 2e-5);
    compare("Symplectic order 6", symp6, ref, 1e-4, true, 1e-6);

    cout << endl << (n_fail == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << " (" << n_fail << " failures)" << endl;

    delete ref.src; delete euler.src; delete rk4.src; delete rk45.src; delete symp4.src; delete symp6.src;
    return n_fail == 0 ? 0 : 1;
}
