/*
 * symplectic_unit_test.cpp
 *
 * Unit tests for the Mino-time helpers in include/kerr.h that underpin the
 * symplectic integrator (Integrator::Symplectic).  These validate the physics
 * mapping between the canonical Mino-time state (u, p_u, theta, p_theta) and
 * the Boyer-Lindquist constants-of-motion description used by the rest of the
 * code, before the propagator itself is built on top of them.
 *
 * Checks
 * ------
 *  (a) Round trip: for every ray of a PointSource grid (several spins, including
 *      negative spin as used by ImagePlane backward tracing), mino_init() followed
 *      by mino_to_bl() reproduces the contravariant momenta from momentum_from_consts().
 *      This also verifies dt/dlambda_M = rhosq*pt and dphi/dlambda_M = rhosq*pphi.
 *  (b) Null condition H_r + H_theta = 0 and carter_Q() == Q at initialisation.
 *  (c) Potential derivatives agree with central finite differences of H_r, H_theta.
 *  (d) Convergence order of the kick/drift composition built from the helpers:
 *      error ratios on halving the step of ~4 (Verlet) and ~16 (Yoshida 4th order).
 *  (e) Time reversibility: N steps forward then N steps backward returns the
 *      initial state to round-off.
 *  (f) (a) and (b) repeated in single precision with looser tolerances.
 *  (g) Exact great-circle polar flow (mino_polar_flow) agrees with brute-force integration,
 *      and keeps a near-axis lamppost ray accurate at the default step.
 *
 * Exit status is 0 if all checks pass, 1 otherwise.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
using namespace std;

#include "raytracer/pointsource.h"
#include "include/kerr.h"

static int n_fail = 0;

static void check(bool ok, const string& what)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << endl;
    if (!ok) ++n_fail;
}

// ---------------------------------------------------------------------------
// Minimal kick/drift steppers built from the kerr.h helpers.  This is the
// reference implementation the propagator will follow.
// ---------------------------------------------------------------------------
template <typename T>
struct MinoState { T u, pu, theta, ptheta, t, phi; };

template <typename T>
static void mino_kick(MinoState<T>& s, T h, T k, T hh, T a)
{
    T r, sqrt_delta;
    mino_r_from_u(s.u, a, r, sqrt_delta);
    s.pu     -= h * mino_dVr_dr(r, k, hh, a) * sqrt_delta;   // dV_r/du = dV_r/dr * dr/du
    s.ptheta -= h * mino_dW_dtheta(s.theta, k, a);          // only the bounded part W of V_theta
    s.t      += h * mino_tdot(r, s.theta, k, hh, a);
    s.phi    += h * (mino_phidot(r, s.theta, k, hh, a) - hh/(sin(s.theta)*sin(s.theta)));   // h/sin^2 part is in the polar flow
}

template <typename T>
static void mino_drift(MinoState<T>& s, T h, T hh)
{
    s.u += h * s.pu;                                   // radial: free drift in u
    mino_polar_flow(s.theta, s.ptheta, s.phi, hh, h);  // polar: exact great-circle flow
}

template <typename T>
static void mino_verlet(MinoState<T>& s, T h, T k, T hh, T a)
{
    mino_kick(s, h/2, k, hh, a);
    mino_drift(s, h, hh);
    mino_kick(s, h/2, k, hh, a);
}

template <typename T>
static void mino_yoshida4(MinoState<T>& s, T h, T k, T hh, T a)
{
    const T c  = cbrt(T(2));
    const T w1 = 1 / (2 - c);
    const T w0 = -c / (2 - c);
    mino_verlet(s, w1*h, k, hh, a);
    mino_verlet(s, w0*h, k, hh, a);
    mino_verlet(s, w1*h, k, hh, a);
}

// ---------------------------------------------------------------------------
template <typename T>
static void round_trip_tests(T spin, T tol, const char* label)
{
    T source[] = {0.0, 5.0, 1e-3, 0.0};
    const T V = 0;
    PointSource<T> src(source, V, spin, TOL, T(0.1), T(0.2));

    const int n = src.get_count();
    T max_rel = 0, max_H = 0, max_Q = 0;
    int n_checked = 0;

    for (int i = 0; i < n; i++)
    {
        const Ray<T>& ray = src.rays[i];
        if (ray.steps < 0) continue;

        // reference contravariant momenta from the existing code path
        T pt0, pr0, pth0, pphi0;
        momentum_from_consts<T>(pt0, pr0, pth0, pphi0, ray.k, ray.h, ray.Q,
                                ray.rdot_sign, ray.thetadot_sign,
                                ray.r, ray.theta, ray.phi, spin);

        // Mino-time canonical state and back
        T u, pu, ptheta;
        mino_init<T>(ray.r, ray.theta, ray.k, ray.h, ray.Q, ray.rdot_sign, ray.thetadot_sign, spin, u, pu, ptheta);
        T r1, pt1, pr1, pth1, pphi1;
        mino_to_bl<T>(u, pu, ray.theta, ptheta, ray.k, ray.h, spin, r1, pt1, pr1, pth1, pphi1);

        auto rel = [](T x, T y) { return abs(x - y) / max(T(1), abs(y)); };
        T e = max(max(rel(r1, ray.r), rel(pt1, pt0)), max(max(rel(pr1, pr0), rel(pth1, pth0)), rel(pphi1, pphi0)));
        if (e > max_rel) max_rel = e;

        // null condition and Carter constant
        T HM = mino_hamiltonian_r<T>(u, pu, ray.k, ray.h, spin) + mino_hamiltonian_theta<T>(ray.theta, ptheta, ray.k, ray.h, spin);
        T Hscale = abs(mino_hamiltonian_theta<T>(ray.theta, ptheta, ray.k, ray.h, spin));
        if (abs(HM) / max(T(1), Hscale) > max_H) max_H = abs(HM) / max(T(1), Hscale);
        T dQ = abs(carter_Q<T>(ray.theta, ptheta, ray.k, ray.h, spin) - ray.Q) / max(T(1), abs(ray.Q));
        if (dQ > max_Q) max_Q = dQ;
        ++n_checked;
    }

    cout << label << ": spin = " << spin << ", " << n_checked << " rays;  max rel momentum diff = "
         << scientific << setprecision(2) << max_rel << ",  max |H_M| = " << max_H << ",  max |dQ| = " << max_Q << endl;
    check(max_rel < tol, "(a) mino_init/mino_to_bl round trip matches momentum_from_consts");
    check(max_H < tol,   "(b) H_r + H_theta = 0 at initialisation");
    check(max_Q < tol,   "(b) carter_Q recovers Q");
}

// ---------------------------------------------------------------------------
static void derivative_tests()
{
    const double a = 0.998, k = 1.0, h = 1.7;
    const double eps = 1e-6;
    double max_err = 0;
    for (double r = 1.5; r < 50; r *= 1.5)
    {
        double u = mino_u_from_r(r, a);
        // d/du of H_r at fixed p_u, by central difference, vs dV_r/dr * dr/du
        double Hp = mino_hamiltonian_r(u + eps, 0.0, k, h, a);
        double Hm = mino_hamiltonian_r(u - eps, 0.0, k, h, a);
        double rr, sd; mino_r_from_u(u, a, rr, sd);
        double analytic = mino_dVr_dr(rr, k, h, a) * sd;
        double err = abs((Hp - Hm)/(2*eps) - analytic) / max(1.0, abs(analytic));
        if (err > max_err) max_err = err;
    }
    for (double th = 0.2; th < M_PI - 0.2; th += 0.3)
    {
        double Hp = mino_hamiltonian_theta(th + eps, 0.0, k, h, a);
        double Hm = mino_hamiltonian_theta(th - eps, 0.0, k, h, a);
        double analytic = mino_dVtheta_dtheta(th, k, h, a);
        double err = abs((Hp - Hm)/(2*eps) - analytic) / max(1.0, abs(analytic));
        if (err > max_err) max_err = err;
    }
    cout << "derivatives: max rel error vs finite difference = " << scientific << setprecision(2) << max_err << endl;
    check(max_err < 1e-6, "(c) potential derivatives match finite differences of H_r, H_theta");
}

// ---------------------------------------------------------------------------
static MinoState<double> test_ray_state(double& k, double& h, double a)
{
    // a tangentially emitted ray (cos alpha = 0, beta = 0.3) from a point source at r = 5,
    // theta = 0.6 -- away from the polar axis, where the h^2/sin^2(theta) barrier would need
    // a much smaller step -- that stays in the strong-field region for the short Mino-time
    // interval integrated here
    double source[] = {0.0, 5.0, 0.6, 0.0};
    PointSource<double> src(source, 0.0, a, TOL, 0.1, 0.2, 0.0, 0.05, 0.3, 0.35);
    const Ray<double>& ray = src.rays[0];
    k = ray.k; h = ray.h;
    MinoState<double> s;
    mino_init<double>(ray.r, ray.theta, ray.k, ray.h, ray.Q, ray.rdot_sign, ray.thetadot_sign, a, s.u, s.pu, s.ptheta);
    s.theta = ray.theta; s.t = 0; s.phi = 0;
    return s;
}

static void convergence_tests()
{
    const double a = 0.998;
    double k, h;
    MinoState<double> s0 = test_ray_state(k, h, a);
    const double lam = 0.15;

    // reference: very fine Yoshida run
    MinoState<double> ref = s0;
    { const int N = 25600; const double dh = lam / N; for (int i = 0; i < N; i++) mino_yoshida4(ref, dh, k, h, a); }

    for (int order = 2; order <= 4; order += 2)
    {
        vector<double> errs;
        // coarser steps for order 4 so the errors stay well above round-off
        const int N0 = (order == 2) ? 100 : 25;
        for (int N = N0; N <= 8*N0; N *= 2)
        {
            MinoState<double> s = s0;
            const double dh = lam / N;
            for (int i = 0; i < N; i++)
                if (order == 2) mino_verlet(s, dh, k, h, a); else mino_yoshida4(s, dh, k, h, a);
            double e = sqrt(pow(s.u - ref.u, 2) + pow(s.theta - ref.theta, 2) + pow(s.phi - ref.phi, 2) + pow((s.t - ref.t)/100, 2));
            errs.push_back(e);
        }
        cout << "order " << order << " errors (N=" << N0 << "," << 2*N0 << "," << 4*N0 << "," << 8*N0 << "): ";
        { double rr, sd; mino_r_from_u(ref.u, a, rr, sd); if (order == 2) cout << "[ref end r = " << fixed << setprecision(3) << rr << "] "; }
        for (double e : errs) cout << scientific << setprecision(2) << e << " ";
        cout << " ratios: ";
        bool ok = true;
        for (size_t i = 1; i < errs.size(); i++)
        {
            double ratio = errs[i-1] / errs[i];
            cout << fixed << setprecision(1) << ratio << " ";
            const double expected = (order == 2) ? 4 : 16;
            if (!(ratio >= 0.6*expected) || !isfinite(errs[i])) ok = false;   // written so that NaN fails
        }
        cout << endl;
        check(ok, string("(d) convergence order ") + (order == 2 ? "2 (Verlet)" : "4 (Yoshida)"));
    }
}

static void reversibility_test()
{
    const double a = 0.998;
    double k, h;
    MinoState<double> s0 = test_ray_state(k, h, a);
    MinoState<double> s = s0;
    const int N = 2000; const double dh = 1e-4;
    for (int i = 0; i < N; i++) mino_yoshida4(s, dh, k, h, a);
    for (int i = 0; i < N; i++) mino_yoshida4(s, -dh, k, h, a);
    double e = max(max(abs(s.u - s0.u), abs(s.pu - s0.pu)), max(max(abs(s.theta - s0.theta), abs(s.ptheta - s0.ptheta)), max(abs(s.t), abs(s.phi))));
    cout << "reversibility: max |state - initial| after forward/backward = " << scientific << setprecision(2) << e << endl;
    check(e < 1e-9, "(e) forward/backward integration returns to the initial state");
}

// ---------------------------------------------------------------------------
static void polar_flow_tests()
{
    // (g1) exactness: the great-circle flow conserves H_c and agrees with a fine Verlet
    //      integration of the same H_c system
    const double h = 0.7, lam = 3.0;
    double th = 0.3, pth = 2.0, phi = 0.0;
    const double Hc0 = 0.5*pth*pth + h*h/(2*sin(th)*sin(th));
    double th_e = th, pth_e = pth, phi_e = phi;
    mino_polar_flow(th_e, pth_e, phi_e, h, lam);
    // brute-force reference: plain Verlet on H_c with a tiny step
    const int N = 300000; const double dh = lam / N;
    for (int i = 0; i < N; i++)
    {
        pth += 0.5*dh * h*h*cos(th)/pow(sin(th), 3); phi += 0.5*dh * h/(sin(th)*sin(th));
        th  += dh * pth;
        pth += 0.5*dh * h*h*cos(th)/pow(sin(th), 3); phi += 0.5*dh * h/(sin(th)*sin(th));
    }
    const double Hc1 = 0.5*pth_e*pth_e + h*h/(2*sin(th_e)*sin(th_e));
    double e = max(max(abs(th_e - th), abs(pth_e - pth)), abs(phi_e - phi));
    cout << "polar flow: |exact - fine Verlet| = " << scientific << setprecision(2) << e
         << ",  |dH_c| = " << abs(Hc1 - Hc0) << endl;
    check(e < 1e-7 && abs(Hc1 - Hc0) < 1e-12, "(g1) exact great-circle polar flow");

    // (g2) near-axis start: lamppost at theta = 1e-3 with h != 0, integrated with h0 = 0.01,
    //      compared with a run at h0 = 0.0005.  With the barrier handled exactly the coarse
    //      step must remain accurate (this case diverges with the naive kick/drift split).
    const double a = 0.998;
    double source[] = {0.0, 5.0, 1e-3, 0.0};
    PointSource<double> src(source, 0.0, a, TOL, 0.1, 0.2, 0.0, 0.05, 0.3, 0.35);
    const Ray<double>& ray = src.rays[0];
    MinoState<double> s0;
    mino_init<double>(ray.r, ray.theta, ray.k, ray.h, ray.Q, ray.rdot_sign, ray.thetadot_sign, a, s0.u, s0.pu, s0.ptheta);
    s0.theta = ray.theta; s0.t = 0; s0.phi = 0;
    MinoState<double> coarse = s0, fine = s0;
    const double lam2 = 0.1;
    for (int i = 0; i < 10; i++)   mino_yoshida4(coarse, lam2/10,   ray.k, ray.h, a);
    for (int i = 0; i < 2000; i++) mino_yoshida4(fine,   lam2/2000, ray.k, ray.h, a);
    double e2 = max(max(abs(coarse.u - fine.u), abs(coarse.theta - fine.theta)), abs(coarse.phi - fine.phi));
    double HM = mino_hamiltonian_r(coarse.u, coarse.pu, ray.k, ray.h, a) + mino_hamiltonian_theta(coarse.theta, coarse.ptheta, ray.k, ray.h, a);
    cout << "near-axis start (theta0 = 1e-3, h = " << scientific << setprecision(2) << ray.h << "): h0 = 0.01 vs 0.0005 difference = "
         << e2 << ",  |H_M| = " << abs(HM) << ",  theta reached " << fixed << setprecision(4) << coarse.theta << endl;
    check(e2 < 1e-6 && abs(HM) < 1e-6, "(g2) near-axis lamppost ray stable at h0 = 0.01");
}

// ---------------------------------------------------------------------------
int main()
{
    cout << "=== Mino-time helper unit tests ===" << endl;

    round_trip_tests<double>(0.998, 1e-9, "double");
    round_trip_tests<double>(0.5,   1e-9, "double");
    round_trip_tests<double>(0.0,   1e-9, "double");
    round_trip_tests<double>(-0.998, 1e-9, "double (negative spin)");

    derivative_tests();
    convergence_tests();
    reversibility_test();
    polar_flow_tests();

    round_trip_tests<float>(0.998f, 2e-3f, "(f) float");
    round_trip_tests<float>(-0.5f,  2e-3f, "(f) float");

    cout << endl << (n_fail == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << " (" << n_fail << " failures)" << endl;
    return n_fail == 0 ? 0 : 1;
}
