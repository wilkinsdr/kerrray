/*
 * photon_ring_test.cpp
 *
 * Photon-ring stress test (T4 of docs/plan_symplectic_integrator.md).
 *
 * Rays are set up directly from constants of motion offset from the analytic critical curve
 * of a Kerr black hole (Bardeen 1973):
 *   lambda_c(r_ph) = -(r^3 - 3r^2 + a^2 r + a^2) / (a (r - 1))
 *   eta_c(r_ph)    = -r^3 (r^3 - 6r^2 + 9r - 4a^2) / (a^2 (r - 1)^2)
 * with h = lambda_c (1 + eps), Q = eta_c, k = 1, for several photon-orbit radii r_ph and offsets
 * eps = +/-1e-2 ... 1e-8, started at r = 1000, theta = 60 deg, ingoing, with both initial
 * polar directions.  Such rays whirl ~ln(1/eps)/(2 pi gamma) times around the photon sphere
 * before escaping or being captured -- the regime that decides winding numbers, higher-order
 * images and caustics, and where the long-term error behaviour of an integrator matters.
 *
 * Each method is compared with a fine symplectic reference (order 6, h0 = 5e-4, max_tstep = 0.05,
 * constant cap) on:
 *   - fate (escape to r_lim vs capture) and the number of equatorial crossings (winding number)
 *   - exit direction (theta, phi) for escaping rays, by eps bin
 *   - the fraction of rays stopped by the step limit, and wall time
 *   - for symplectic runs: |H_M| and |dQ|/Q recomputed from the final state
 * For the smallest eps the orbits are chaotic enough that even the reference exit direction is
 * uncertain; fate and crossing count are the robust quantities there.
 *
 * Chaos sets the floor.  Photon-sphere orbits are unstable with a Lyapunov exponent of order
 * e^6 per orbit (Schwarzschild: gamma T_orbit = 2 pi 3 sqrt(3) / sqrt(27) ~ 6.3), so a ray that
 * whirls n times amplifies any error by ~500^n: beyond ~2-3 orbits (eps <~ 1e-4) no double
 * precision integrator reproduces another's exit direction, fate or winding number ray by ray.
 * To make that floor visible the reference is also compared with a second reference run at a
 * slightly different step ("reference self-consistency"); agreement rates at small eps should be
 * read against that row, not against 100%.
 *
 * Pass criteria (symplectic runs):
 *   - no step-limit failures
 *   - fate and crossing-count agreement 100% for eps >= 1e-3 (<~ 1-2 orbits, deterministic regime)
 *   - max |H_M| < 1e-6
 * RK45 results are reported for comparison (informational; a ray stopped by its step limit is
 * counted as captured, since that is what those rays are).
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <map>
using namespace std;

#include "raytracer/raytracer.h"
#include "include/kerr.h"

static const double SPIN   = 0.998;
static const double R0     = 1000.0;
static const double THETA0 = 60.0 * M_PI / 180.0;
static const double R_LIM  = 1500.0;

struct RaySpec { double rph, eps; int tsign; double h, Q; };

static vector<RaySpec> make_specs()
{
    vector<RaySpec> v;
    const double a = SPIN;
    for (double rph : {1.5, 2.0, 2.5, 3.0, 3.5})
    {
        const double lc = -(rph*rph*rph - 3*rph*rph + a*a*rph + a*a) / (a*(rph - 1));
        const double ec = -rph*rph*rph*(rph*rph*rph - 6*rph*rph + 9*rph - 4*a*a) / (a*a*(rph - 1)*(rph - 1));
        for (int e = 2; e <= 8; e++)
            for (int sgn : {+1, -1})
                for (int tsign : {+1, -1})
                {
                    const double eps = sgn * pow(10.0, -e);
                    const double h = lc * (1 + eps), Q = ec;
                    // polar motion must be allowed at theta0
                    const double c = cos(THETA0), s = sin(THETA0);
                    if (Q + c*c*(a*a - h*h/(s*s)) <= 0) continue;
                    v.push_back({rph, eps, tsign, h, Q});
                }
    }
    return v;
}

static Raytracer<double>* setup(const vector<RaySpec>& specs)
{
    Raytracer<double>* rt = new Raytracer<double>((int)specs.size(), SPIN, PRECISION);
    for (size_t i = 0; i < specs.size(); i++)
    {
        Ray<double>& r = rt->rays[i];
        r.t = 0; r.r = R0; r.theta = THETA0; r.phi = 0;
        r.pt = r.pr = r.ptheta = r.pphi = 0;
        r.k = 1.0; r.h = specs[i].h; r.Q = specs[i].Q;
        r.rdot_sign = -1; r.thetadot_sign = specs[i].tsign;
        r.rdot_flips = 0; r.equatorial_crossings = 0;
        r.steps = 0; r.status = 0; r.emit = 1; r.redshift = 0;
        r.alpha = specs[i].eps; r.beta = specs[i].rph;
    }
    return rt;
}

struct Result { Raytracer<double>* rt; double seconds; };

static Result run(const vector<RaySpec>& specs, Integrator method, double tol, int order, double h0, double mdt, double rlim_cap)
{
    Raytracer<double>* rt = setup(specs);
    rt->set_rk45_tol(tol);
    rt->set_symplectic_order(order);
    rt->set_symplectic_step(h0);
    rt->set_max_tstep(mdt, rlim_cap);
    auto t0 = chrono::high_resolution_clock::now();
    rt->run_raytrace(method, 0.0, R_LIM, 0);   // thetalim = 0: no polar stopping condition
    auto t1 = chrono::high_resolution_clock::now();
    return { rt, chrono::duration<double>(t1 - t0).count() };
}

static int fate(const Ray<double>& r)
{
    if (r.status & RAY_STATUS_STEPLIM) return 1;   // stalled at the step limit: a plunging ray in practice
    if (r.status & RAY_STATUS_HORIZON) return 1;
    return 0;   // escaped to R_LIM
}

static int n_fail = 0;
static void check(bool ok, const string& what)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << endl;
    if (!ok) ++n_fail;
}

static void report(const string& name, const vector<RaySpec>& specs, const Result& res, const Raytracer<double>* ref, bool symplectic, bool judge)
{
    const int n = (int)specs.size();
    int n_steplim = 0; long steps = 0; double max_H = 0, max_dQ = 0;
    // per-eps-decade statistics
    map<int, int> n_bin, fate_ok, cross_ok, esc_both;
    map<int, vector<double>> dth, dph;
    for (int i = 0; i < n; i++)
    {
        const Ray<double>& a = res.rt->rays[i];
        const Ray<double>& b = ref->rays[i];
        const int e = (int)lround(-log10(abs(specs[i].eps)));
        steps += abs(a.steps);
        if (a.status & RAY_STATUS_STEPLIM) ++n_steplim;
        ++n_bin[e];
        if (fate(a) == fate(b)) ++fate_ok[e];
        if (fate(a) == fate(b) && a.equatorial_crossings == b.equatorial_crossings) ++cross_ok[e];
        if (fate(a) == 0 && fate(b) == 0)
        {
            ++esc_both[e];
            dth[e].push_back(abs(a.theta - b.theta));
            double d = a.phi - b.phi; dph[e].push_back(abs(atan2(sin(d), cos(d))));
        }
        if (symplectic && fate(a) == 0)
        {
            const double rhosq = a.r*a.r + (SPIN*cos(a.theta))*(SPIN*cos(a.theta));
            const double u = mino_u_from_r(a.r, SPIN); double rr, sd; mino_r_from_u(u, SPIN, rr, sd);
            const double pu = rhosq*a.pr/sd, pth = rhosq*a.ptheta;
            const double HM = mino_hamiltonian_r(u, pu, a.k, a.h, SPIN) + mino_hamiltonian_theta(a.theta, pth, a.k, a.h, SPIN);
            max_H = max(max_H, abs(HM) / max(1.0, abs(mino_hamiltonian_theta(a.theta, pth, a.k, a.h, SPIN))));
            max_dQ = max(max_dQ, abs(carter_Q(a.theta, pth, a.k, a.h, SPIN) - a.Q) / max(1.0, abs(a.Q)));
        }
    }
    auto med = [](vector<double> v) -> double { if (v.empty()) return NAN; sort(v.begin(), v.end()); return v[v.size()/2]; };
    auto mx  = [](vector<double> v) -> double { if (v.empty()) return NAN; return *max_element(v.begin(), v.end()); };

    cout << endl << "=== " << name << " ===" << endl;
    cout << fixed << setprecision(3) << "  time " << res.seconds << " s, mean steps/ray " << (double)steps / n
         << ", step-limit failures " << n_steplim << " / " << n;
    if (symplectic) cout << scientific << setprecision(1) << ", max |H_M| " << max_H << ", max |dQ|/Q " << max_dQ;
    cout << endl;
    cout << "  eps     fate agree   crossings agree   escaped(both)   median|dtheta|  max|dtheta|   median|dphi|   max|dphi|" << endl;
    int fate_ok_3 = 0, n_3 = 0, cross_ok_3 = 0;
    for (auto& kv : n_bin)
    {
        const int e = kv.first;
        cout << "  1e-" << e << "     " << setw(3) << fate_ok[e] << " / " << setw(3) << kv.second
             << "        " << setw(3) << cross_ok[e] << " / " << setw(3) << kv.second
             << "          " << setw(3) << esc_both[e]
             << scientific << setprecision(2) << "        " << med(dth[e]) << "      " << mx(dth[e]) << "      " << med(dph[e]) << "     " << mx(dph[e]) << endl;
        if (e <= 3) { fate_ok_3 += fate_ok[e]; cross_ok_3 += cross_ok[e]; n_3 += kv.second; }
    }
    if (judge)
    {
        check(n_steplim == 0, name + ": no step-limit failures");
        check(fate_ok_3 == n_3 && cross_ok_3 == n_3, name + ": fate and crossing-count agreement 100% for eps >= 1e-3");
        check(max_H < 1e-6, name + ": max |H_M| < 1e-6");
    }
}

int main()
{
    vector<RaySpec> specs = make_specs();
    cout << "photon-ring stress test: spin " << SPIN << ", " << specs.size() << " rays from r = " << R0
         << ", theta = 60 deg, r_ph in {1.5, 2, 2.5, 3, 3.5}, eps = +/-1e-2 .. 1e-8" << endl;

    Result ref = run(specs, Integrator::Symplectic, 1e-8, 6, 5e-4, 0.05, 1e12);
    {
        int esc = 0, cap = 0, sl = 0; int maxc = 0;
        for (size_t i = 0; i < specs.size(); i++) { int f = fate(ref.rt->rays[i]); esc += (f == 0); cap += (f == 1); sl += (f == 2); maxc = max(maxc, ref.rt->rays[i].equatorial_crossings); }
        cout << "reference (symplectic o6, h0 5e-4, max_tstep 0.05): " << fixed << setprecision(2) << ref.seconds << " s; escaped " << esc
             << ", captured " << cap << ", step limit " << sl << ", max equatorial crossings " << maxc << endl;
    }

    Result refB  = run(specs, Integrator::Symplectic, 1e-8, 6, 4e-4, 0.04, 1e12);   // second reference: chaos floor
    Result rk45a = run(specs, Integrator::RK45, 1e-8,  6, -1, MAXDT, MAXDT_RLIM);
    Result rk45b = run(specs, Integrator::RK45, 1e-10, 6, -1, MAXDT, MAXDT_RLIM);
    Result symd  = run(specs, Integrator::Symplectic, 1e-8, 6, -1,     MAXDT, MAXDT_RLIM);   // defaults: o6, h0 = 1/precision, max_tstep 1, growth
    Result symt  = run(specs, Integrator::Symplectic, 1e-8, 6, 0.0025, 0.25,  MAXDT_RLIM);   // tightened

    report("Reference self-consistency (o6, h0 4e-4, max_tstep 0.04 vs h0 5e-4, 0.05)", specs, refB, ref.rt, true, false);
    report("RK45 tol 1e-8",  specs, rk45a, ref.rt, false, false);
    report("RK45 tol 1e-10", specs, rk45b, ref.rt, false, false);
    report("Symplectic default (o6, h0 0.01, max_tstep 1)", specs, symd, ref.rt, true, true);
    report("Symplectic tight (o6, h0 0.0025, max_tstep 0.25)", specs, symt, ref.rt, true, true);

    cout << endl << (n_fail == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << " (" << n_fail << " failures)" << endl;
    delete ref.rt; delete refB.rt; delete rk45a.rt; delete rk45b.rt; delete symd.rt; delete symt.rt;
    return n_fail == 0 ? 0 : 1;
}
