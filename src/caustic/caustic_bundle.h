/*
 * caustic_bundle.h
 *
 *  CausticBundle: the lockstep ray-bundle tracer used by caustic_3d and caustic_ent to locate the caustics of
 *  the family of rays reaching an observer.  A bundle is built from one ImagePlane at an image-plane position
 *  (x, y): the centre ray and four offset rays at (x +/- delta, y), (x, y +/- delta), initialised with
 *  ImagePlane::init_ray() and converted to the canonical Mino-time state (mino_init, kerr.h).  trace() advances
 *  the five rays with identical Mino-time steps (MinoStepper, mino_stepper.h; step size from mino_step_size())
 *  and forms the Jacobian of the ray family
 *
 *      J(tau) = det[ dX/dx , dX/dy , dX/dtau ]
 *
 *  from central differences of the Cartesian positions of the offset rays and the Mino-time velocity of the
 *  centre ray.  Sign changes of J along the centre ray are caustic crossings, located by bisection on the
 *  fraction of the step (the composed step is re-taken for all five rays from the saved previous states) and
 *  stored as CausticPoints.  With Params::max_eqcross > 0 the bundle is also landed exactly on theta = pi/2 at
 *  the centre ray's equatorial crossings and J, the state and the constants of motion recorded there
 *  (DiscCrossing; used by caustic_ent, where the caustic meets the disc wherever this J vanishes).
 *
 *  Only sign(J) matters, so the bundle is kept in the linear regime through photon-shell orbits by rescaling
 *  each offset pair's deviation from the centre ray (in the full state and in h) back to delta whenever it
 *  exceeds delta_max; the accumulated log10 scale factor is recorded with every result.
 *
 *  Verified against the previous free-function implementation (caustic_bundle_regression_test) and in the
 *  Schwarzschild limit (caustic_3d_test); see docs/plan_caustic_bundle_class.md.
 */

#ifndef CAUSTIC_BUNDLE_H_
#define CAUSTIC_BUNDLE_H_

#include <cmath>
#include <vector>
#include <ostream>
#include <iomanip>
#include <limits>
#include <algorithm>

#include "../raytracer/raytracer.h"
#include "../raytracer/imageplane.h"
#include "../raytracer/mino_stepper.h"
#include "../include/kerr.h"

// The bundle is stopped once the centre ray reaches u < BUNDLE_U_STOP in the radial coordinate of the
// Mino-time stepper (r = 1 + b cosh u, b = sqrt(1 - a^2); u = 0 is the horizon and u = 0.25 is
// r - r+ = 0.03 b).  Within that distance of the horizon the finite differences between the bundle rays
// are no longer trustworthy (the radial potential is singular at u = 0 and the composed step's backward
// substeps approach it), producing spurious sign changes of J on the last step before capture.
#define BUNDLE_U_STOP 0.25

// bundle termination status codes (STATUS extension of the FITS outputs)
enum BundleStatus { STATUS_ESCAPED = 0, STATUS_HORIZON = 1, STATUS_STEPLIM = 2, STATUS_SPLIT = 3,
                    STATUS_SKIPPED = 4, STATUS_NAN = 5, STATUS_EQSTOP = 6 };

// a caustic crossing of the centre ray
struct CausticPoint
{
    int ix, iy, n;             // image-plane point labels and crossing index along the ray (1-based)
    double ximg, yimg;
    double tau, t, r, theta, phi;
    double X, Y, Z;
    int rflips, eqcross, dj_sign;
    double logscale;
    int level = 0;             // refinement level of the image-plane point (adaptive_plane.h); 0 = regular grid
};

// the state of the bundle when the centre ray crosses the equatorial plane (theta = pi/2)
//   J        finite-difference Jacobian of the (rescaled) bundle at the crossing; J * 10^logscale is the
//            Jacobian of the unscaled family, continuous from point to point
//   ncaust   caustic crossings along the ray before this equatorial crossing
struct DiscCrossing
{
    int n;                     // equatorial crossing index (1-based)
    double J, logscale;
    double tau, t, r, phi;     // centre ray at the crossing (phi wrapped to (-pi, pi])
    double u, pu, ptheta;      // canonical Mino-time state of the centre ray (theta = pi/2)
    double k, h;               // constants of motion of the centre ray
    int ncaust, rflips;
};

// per-bundle summary (the per-pixel maps of the applications)
struct PixelResult
{
    int status;
    int ncaust;
    int eqcross;
    double tau_end;
    double r1, theta1, phi1;   // first caustic crossing (0 if none)
};

template <typename T = double>
class CausticBundle
{
public:
    // tracing parameters, shared by all bundles of a run
    struct Params
    {
        T a;                   // spin used for the propagation (negated for backward tracing)
        T horizon;
        T h0;                  // Mino-time step in the strong field
        T r_cap;               // far-field step caps apply outside this radius
        T max_tstep, maxtstep_rlim, max_phistep;
        T r_max;               // escape radius
        T delta;               // bundle offset in the image plane (rg)
        T delta_max;           // rescale the bundle when an offset ray is further than this from the centre
        int order;             // composition order of the symplectic stepper
        int steplim;
        int max_caustics;      // caustic points stored per bundle (all are counted)
        int max_eqcross = 0;   // record the bundle state at the first max_eqcross equatorial crossings (0: none)
        int stop_after_eqcross = 0;  // stop the bundle after this many equatorial crossings (0: never)

        // defaults for bundles started from plane (spin, horizon and the far-field cap radius)
        static Params from_plane(const ImagePlane<T>& plane, T spin, T delta, T symp_step, T precision)
        {
            Params P;
            P.a = -spin;                           // ImagePlane traces backwards in time by negating the spin
            P.horizon = kerr_horizon<T>(spin);
            P.h0 = (symp_step > 0) ? symp_step : T(1) / precision;
            P.r_cap = 2 * P.horizon;
            P.max_tstep = MAXDT; P.maxtstep_rlim = MAXDT_RLIM; P.max_phistep = MAXDPHI;
            P.r_max = T(1.1) * plane.get_dist();
            P.delta = delta; P.delta_max = 100 * delta;
            P.order = 6; P.steplim = SYMP_STEPLIM; P.max_caustics = 32;
            return P;
        }
    };

    // Build the bundle centred on image-plane position (x, y) of plane.  ix, iy, level label the point in
    // the results (regular-grid indices, or finest-grid indices and level of an adaptively refined point).
    CausticBundle(const ImagePlane<T>& plane, const Params& P, T x, T y, int ix = 0, int iy = 0, int level = 0)
        : m_plane(plane), m_P(P), m_x(x), m_y(y), m_ix(ix), m_iy(iy), m_level(level),
          m_st{ MinoStepper<T>(P.a, 0, 0, P.order), MinoStepper<T>(P.a, 0, 0, P.order), MinoStepper<T>(P.a, 0, 0, P.order),
                MinoStepper<T>(P.a, 0, 0, P.order), MinoStepper<T>(P.a, 0, 0, P.order) }
    {
        const T offx[5] = { 0, P.delta, -P.delta, 0, 0 };
        const T offy[5] = { 0, 0, 0, P.delta, -P.delta };
        m_valid = true;
        for (int i = 0; i < 5; i++)
        {
            Ray<T> ray;
            plane.init_ray(ray, x + offx[i], y + offy[i]);
            if (i == 0) m_centre_init = ray;
            m_ray[i].k = ray.k;
            m_ray[i].h = ray.h;
            m_ray[i].s.t = ray.t;
            m_ray[i].s.theta = ray.theta;
            m_ray[i].s.phi = ray.phi;
            if (!std::isfinite(ray.h) || !std::isfinite(ray.Q) || !std::isfinite(ray.theta) || !std::isfinite(ray.phi)) m_valid = false;
            mino_init<T>(ray.r, ray.theta, ray.k, ray.h, ray.Q, ray.rdot_sign, ray.thetadot_sign, P.a,
                         m_ray[i].s.u, m_ray[i].s.pu, m_ray[i].s.ptheta);
            m_st[i] = MinoStepper<T>(P.a, ray.k, ray.h, P.order);
            m_init[i] = m_ray[i];
        }
        m_res.status = m_valid ? STATUS_STEPLIM : STATUS_SKIPPED;
        m_res.ncaust = 0; m_res.eqcross = 0; m_res.tau_end = 0;
        m_res.r1 = m_res.theta1 = m_res.phi1 = 0;
    }

    bool valid() const { return m_valid; }

    // --- results ---
    const PixelResult& result() const { return m_res; }
    int status() const { return m_res.status; }
    int ncaust() const { return m_res.ncaust; }
    int eqcross() const { return m_res.eqcross; }
    T tau_end() const { return m_res.tau_end; }
    const std::vector<CausticPoint>& caustics() const { return m_caustics; }
    const std::vector<DiscCrossing>& disc_crossings() const { return m_disc; }
    const CausticPoint* first_caustic() const { return m_caustics.empty() ? nullptr : &m_caustics.front(); }

    // --- current state ---
    T x() const { return m_x; }
    T y() const { return m_y; }
    int ix() const { return m_ix; }
    int iy() const { return m_iy; }
    int level() const { return m_level; }
    const MinoState<T>& centre() const { return m_ray[0].s; }
    T jacobian() const { return jacobian_of(m_ray); }
    T log_scale() const { return m_logscale; }

    // observer-frame energy of the centre ray at the image plane (Raytracer::redshift_start(0, true))
    T emit() const { return m_plane.emit_energy(m_centre_init, T(0), true); }

    // E_disc / E_obs for a photon emitted from a circular Keplerian orbit (V = -1) at an equatorial crossing of
    // the centre ray, with Raytracer::ray_redshift in reverse mode as for ImagePlane rays.  The momentum
    // signs are those of the backward-traced ray at the crossing (dr/dlambda ~ p_u since r increases with u).
    T disc_redshift(const DiscCrossing& dc) const
    {
        const T Q = carter_Q<T>(M_PI_2, dc.ptheta, dc.k, dc.h, m_P.a);
        const int rdot_sign = (dc.pu >= 0) ? 1 : -1;
        const int thetadot_sign = (dc.ptheta >= 0) ? 1 : -1;
        return m_plane.ray_redshift(T(-1), true, false, dc.r, M_PI_2, dc.phi, dc.k, dc.h, Q,
                                   rdot_sign, thetadot_sign, emit());
    }

    // Trace the bundle from its initial data until the centre ray falls through the horizon, escapes, reaches
    // the step limit or (stop_after_eqcross) its last requested equatorial crossing.  dump: optional per-step
    // record of the centre ray and J (diagnostics).  May be called again (e.g. after changing nothing but the
    // dump stream): the trace restarts from the initial data and gives the same results.
    void trace(std::ostream* dump = nullptr)
    {
        if (!m_valid) return;
        for (int i = 0; i < 5; i++) { m_ray[i] = m_init[i]; m_st[i] = MinoStepper<T>(m_P.a, m_init[i].k, m_init[i].h, m_P.order); }
        m_res.status = STATUS_STEPLIM;
        m_res.ncaust = 0; m_res.eqcross = 0; m_res.tau_end = 0;
        m_res.r1 = m_res.theta1 = m_res.phi1 = 0;
        m_caustics.clear(); m_disc.clear();

        m_rflips = 0; m_eqcross = 0; m_logscale = 0; m_tau = 0; m_steps = 0;
        m_J_prev = jacobian_of(m_ray);

        T sd;
        mino_r_from_u<T>(m_ray[0].s.u, m_P.a, m_r, sd);
        dump_state(dump, 0, m_J_prev);

        while (m_steps < m_P.steplim)
        {
            ++m_steps;

            // step size from the centre ray (same policy as Raytracer::propagate_symplectic)
            T tdot, phidot;
            const T step = mino_step_size<T>(m_r, m_ray[0].s.theta, m_ray[0].k, m_ray[0].h, m_P.a, m_P.h0, m_P.r_cap,
                                             m_P.max_tstep, m_P.maxtstep_rlim, m_P.max_phistep, tdot, phidot);

            for (int i = 0; i < 5; i++) m_prev[i] = m_ray[i];
            const int horizon_ray = step_all(m_ray, step);

            if (horizon_ray == 0 || m_ray[0].s.u < BUNDLE_U_STOP)
            {
                m_res.status = STATUS_HORIZON;
                break;
            }
            bool offset_captured = (horizon_ray > 0);
            for (int i = 1; i < 5; i++)
                if (m_ray[i].s.u < BUNDLE_U_STOP) offset_captured = true;
            if (offset_captured)
            {
                // an offset ray was captured while the centre ray was not: the bundle straddles the
                // critical curve and the finite differences are no longer meaningful
                m_res.status = STATUS_SPLIT;
                break;
            }
            if (!all_finite())
            {
                for (int i = 0; i < 5; i++) m_ray[i] = m_prev[i];
                m_res.status = STATUS_NAN;
                break;
            }

            m_tau += step;
            mino_r_from_u<T>(m_ray[0].s.u, m_P.a, m_r, sd);

            // --- caustic crossing: sign change of J across the step ---
            const T J = jacobian_of(m_ray);
            T caustic_frac = 2*step;   // fraction of the step at which a caustic was crossed (> step: none)
            if (std::isfinite(J) && std::isfinite(m_J_prev) && m_J_prev * J < 0)
            {
                const T J_prev = m_J_prev;
                const T frac = bisect_step(step, [&](int hr) -> bool
                {
                    // crossed if the (partial) step hit the horizon, J is undefined, or J changed sign
                    if (hr >= 0) return true;
                    const T Jm = jacobian_of(m_trial);
                    return !std::isfinite(Jm) || Jm * J_prev < 0;
                });
                caustic_frac = frac;
                record_caustic(frac, step, J);
            }
            m_J_prev = J;
            dump_state(dump, m_steps, J);

            // --- equatorial crossing of the centre ray: land the bundle exactly on theta = pi/2 and record it ---
            const bool eq_now = crossed_equator(m_prev[0].s.theta, m_ray[0].s.theta);
            if (eq_now && m_P.max_eqcross > 0 && m_eqcross < m_P.max_eqcross)
            {
                const T side_prev = m_prev[0].s.theta - M_PI_2;
                const T frac = bisect_step(step, [&](int hr) -> bool
                {
                    const T side = m_trial[0].s.theta - M_PI_2;
                    return hr >= 0 || !std::isfinite(side) || side * side_prev <= 0;
                });
                record_disc_crossing(frac, step, caustic_frac);
            }

            // --- bookkeeping on the centre ray ---
            if (m_prev[0].s.pu * m_ray[0].s.pu < 0) ++m_rflips;
            if (eq_now) ++m_eqcross;

            if (m_P.stop_after_eqcross > 0 && m_eqcross >= m_P.stop_after_eqcross)
            {
                m_res.status = STATUS_EQSTOP;
                break;
            }

            if (m_r >= m_P.r_max)
            {
                m_res.status = STATUS_ESCAPED;
                break;
            }

            // --- keep the bundle in the linear regime: rescale each offset pair when it has grown too far ---
            check_rescale();
        }

        m_res.eqcross = m_eqcross;
        m_res.tau_end = m_tau;
    }

private:
    struct Member
    {
        MinoState<T> s;
        T k, h;
    };

    const ImagePlane<T>& m_plane;
    Params m_P;
    T m_x, m_y;
    int m_ix, m_iy, m_level;
    bool m_valid;
    Ray<T> m_centre_init;                  // initial data of the centre ray (for emit())

    Member m_init[5];                           // initial data (trace() restarts from here)
    Member m_ray[5], m_prev[5], m_trial[5];    // b[0] centre, b[1] +x, b[2] -x, b[3] +y, b[4] -y
    MinoStepper<T> m_st[5];

    T m_tau, m_logscale, m_J_prev, m_r;
    int m_steps, m_rflips, m_eqcross;

    PixelResult m_res;
    std::vector<CausticPoint> m_caustics;
    std::vector<DiscCrossing> m_disc;

    static T wrap_phi(T phi) { return atan2(sin(phi), cos(phi)); }

    static bool crossed_equator(T theta_prev, T theta)
    {
        return (theta_prev < M_PI_2 && theta >= M_PI_2) || (theta_prev > M_PI_2 && theta <= M_PI_2);
    }

    // Cartesian position of a bundle ray (kerr.h cartesian())
    void position(const Member& b, T X[3]) const
    {
        T r, sd;
        mino_r_from_u<T>(b.s.u, m_P.a, r, sd);
        cartesian<T>(X[0], X[1], X[2], r, b.s.theta, b.s.phi, m_P.a);
    }

    // Cartesian velocity dX/dlambda_M of a bundle ray
    void velocity(const Member& b, T V[3]) const
    {
        T r, sd;
        mino_r_from_u<T>(b.s.u, m_P.a, r, sd);
        const T rdot  = sd * b.s.pu;                                        // dr/dlambda_M = b sinh(u) p_u
        const T thdot = b.s.ptheta;
        const T phdot = mino_phidot<T>(r, b.s.theta, b.k, b.h, m_P.a);

        const T R  = sqrt(r*r + m_P.a*m_P.a);
        const T dR = r / R;
        const T st = sin(b.s.theta), ct = cos(b.s.theta);
        const T sp = sin(b.s.phi),   cp = cos(b.s.phi);

        V[0] = dR*rdot*st*cp + R*ct*cp*thdot - R*st*sp*phdot;
        V[1] = dR*rdot*st*sp + R*ct*sp*thdot + R*st*cp*phdot;
        V[2] = rdot*ct - r*st*thdot;
    }

    static T det3(const T u[3], const T v[3], const T w[3])
    {
        return u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0]) + u[2]*(v[0]*w[1] - v[1]*w[0]);
    }

    // Jacobian J = det[dX/dx, dX/dy, dX/dtau] of the ray family from a set of five rays
    T jacobian_of(const Member b[5]) const
    {
        T X[5][3];
        for (int i = 0; i < 5; i++) position(b[i], X[i]);
        T xi_x[3], xi_y[3], V[3];
        for (int c = 0; c < 3; c++)
        {
            xi_x[c] = (X[1][c] - X[2][c]) / (2*m_P.delta);
            xi_y[c] = (X[3][c] - X[4][c]) / (2*m_P.delta);
        }
        velocity(b[0], V);
        return det3(xi_x, xi_y, V);
    }

    // advance all five rays of a set by the same Mino-time step; returns the index of the first ray that
    // reached the horizon (or -1)
    int step_all(Member b[5], T step)
    {
        int horizon_ray = -1;
        for (int i = 0; i < 5; i++)
        {
            m_st[i].h = b[i].h;
            m_st[i].step(b[i].s, step);
            if (m_st[i].hit_horizon && horizon_ray < 0) horizon_ray = i;
        }
        return horizon_ray;
    }

    bool all_finite() const
    {
        for (int i = 0; i < 5; i++)
            if (!std::isfinite(m_ray[i].s.u) || !std::isfinite(m_ray[i].s.theta) || !std::isfinite(m_ray[i].s.pu)
                || !std::isfinite(m_ray[i].s.ptheta) || !std::isfinite(m_ray[i].s.phi))
                return false;
        return true;
    }

    // Bisection on the fraction of the last step: re-takes the composed step from m_prev into m_trial with
    // fraction mid and asks crossed(horizon_ray) whether the event lies in (0, mid]; returns the fraction
    // of the event (60 iterations, ~1e-13 of a step) and leaves m_trial at that fraction.
    template <class Pred>
    T bisect_step(T step, Pred crossed)
    {
        T lo = 0, hi = step;
        for (int it = 0; it < 60 && (hi - lo) > 1e-13*step; it++)
        {
            const T mid = (lo + hi) / 2;
            for (int i = 0; i < 5; i++) m_trial[i] = m_prev[i];
            const int hr = step_all(m_trial, mid);
            if (crossed(hr)) hi = mid; else lo = mid;
        }
        const T frac = (lo + hi) / 2;
        for (int i = 0; i < 5; i++) m_trial[i] = m_prev[i];
        step_all(m_trial, frac);
        return frac;
    }

    void record_caustic(T frac, T step, T J_now)
    {
        T rc, sdc;
        mino_r_from_u<T>(m_trial[0].s.u, m_P.a, rc, sdc);

        CausticPoint cp;
        cp.ix = m_ix; cp.iy = m_iy;
        cp.ximg = m_x; cp.yimg = m_y;
        cp.n = m_res.ncaust + 1;
        cp.tau = m_tau - step + frac;
        cp.t = m_trial[0].s.t;
        cp.r = rc;
        cp.theta = m_trial[0].s.theta;
        cp.phi = wrap_phi(m_trial[0].s.phi);
        cartesian<T>(cp.X, cp.Y, cp.Z, rc, m_trial[0].s.theta, m_trial[0].s.phi, m_P.a);
        cp.rflips  = m_rflips  + ((m_prev[0].s.pu * m_trial[0].s.pu < 0) ? 1 : 0);
        cp.eqcross = m_eqcross + (crossed_equator(m_prev[0].s.theta, m_trial[0].s.theta) ? 1 : 0);
        cp.dj_sign = (J_now > 0) ? 1 : -1;
        cp.logscale = m_logscale;
        cp.level = m_level;

        if (m_res.ncaust < m_P.max_caustics) m_caustics.push_back(cp);
        if (m_res.ncaust == 0)
        {
            m_res.r1 = cp.r; m_res.theta1 = cp.theta; m_res.phi1 = cp.phi;
        }
        ++m_res.ncaust;
    }

    void record_disc_crossing(T frac, T step, T caustic_frac)
    {
        DiscCrossing dc;
        dc.n = m_eqcross + 1;
        dc.J = jacobian_of(m_trial);
        dc.logscale = m_logscale;
        dc.tau = m_tau - step + frac;
        dc.t = m_trial[0].s.t;
        T sdc;
        mino_r_from_u<T>(m_trial[0].s.u, m_P.a, dc.r, sdc);
        dc.phi = wrap_phi(m_trial[0].s.phi);
        dc.u = m_trial[0].s.u; dc.pu = m_trial[0].s.pu; dc.ptheta = m_trial[0].s.ptheta;
        dc.k = m_trial[0].k; dc.h = m_trial[0].h;
        // caustics crossed before this point along the ray: those of previous steps plus one if the caustic
        // crossing in this step came earlier than the equatorial crossing
        dc.ncaust = m_res.ncaust - ((caustic_frac <= step && caustic_frac > frac) ? 1 : 0);
        dc.rflips = m_rflips + ((m_prev[0].s.pu * m_trial[0].s.pu < 0) ? 1 : 0);
        m_disc.push_back(dc);
    }

    // rescale the deviation of offset pair (i, j) from the centre ray by s (in the full state and in h)
    void rescale_pair(int i, int j, T s)
    {
        const Member& c = m_ray[0];
        for (int m : {i, j})
        {
            m_ray[m].s.u      = c.s.u      + s*(m_ray[m].s.u      - c.s.u);
            m_ray[m].s.pu     = c.s.pu     + s*(m_ray[m].s.pu     - c.s.pu);
            m_ray[m].s.theta  = c.s.theta  + s*(m_ray[m].s.theta  - c.s.theta);
            m_ray[m].s.ptheta = c.s.ptheta + s*(m_ray[m].s.ptheta - c.s.ptheta);
            m_ray[m].s.t      = c.s.t      + s*(m_ray[m].s.t      - c.s.t);
            // phi is taken modulo 2pi: offset rays passing a pole on opposite sides of the axis end up at
            // the same azimuth with BL phi values differing by 2pi
            m_ray[m].s.phi    = c.s.phi    + s*wrap_phi(m_ray[m].s.phi - c.s.phi);
            m_ray[m].h        = c.h        + s*(m_ray[m].h        - c.h);
        }
    }

    void check_rescale()
    {
        T Xc[3], Xi[3], Xj[3];
        position(m_ray[0], Xc);
        const int pairs[2][2] = { {1, 2}, {3, 4} };
        for (int p = 0; p < 2; p++)
        {
            const int i = pairs[p][0], j = pairs[p][1];
            position(m_ray[i], Xi);
            position(m_ray[j], Xj);
            T sep = 0;
            for (int c = 0; c < 3; c++)
            {
                sep = std::max(sep, std::fabs(Xi[c] - Xc[c]));
                sep = std::max(sep, std::fabs(Xj[c] - Xc[c]));
            }
            if (sep > m_P.delta_max)
            {
                const T s = m_P.delta / sep;
                rescale_pair(i, j, s);
                m_logscale += log10(1/s);
                m_J_prev *= s;   // keep the finite-difference J continuous across the rescaling
            }
        }
    }

    void dump_state(std::ostream* dump, int step_no, T J_now) const
    {
        if (dump == nullptr) return;
        T X[3];
        position(m_ray[0], X);
        (*dump) << step_no << ' ' << std::setprecision(12) << m_tau << ' ' << m_ray[0].s.t << ' ' << m_r << ' '
                << m_ray[0].s.theta << ' ' << m_ray[0].s.phi << ' ' << X[0] << ' ' << X[1] << ' ' << X[2] << ' '
                << J_now << ' ' << m_logscale << ' ' << m_rflips << ' ' << m_eqcross << '\n';
    }
};

#endif /* CAUSTIC_BUNDLE_H_ */
