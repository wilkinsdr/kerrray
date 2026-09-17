/*
 * mino_stepper.h
 *
 *  Symplectic (Mino-time) integration step for null geodesics in the Kerr spacetime.
 *
 *  Integrates the canonical Mino-time system (see the Mino-time helpers in kerr.h):
 *
 *    H_M = 1/2 p_u^2 + 1/2 p_theta^2 + V_r(u) + V_theta(theta) = 0,   d lambda_M = d lambda / rhosq
 *
 *  with r = 1 + b cosh(u), b = sqrt(1 - a^2).  The momenta are integrated directly (no square roots of
 *  R(r) and Theta(theta), so no sign-flip tracking); the constants of motion k and h enter only through
 *  the potentials and Q is conserved to within the truncation error (diagnostic: carter_Q()).
 *
 *  One step is a composition of Stormer-Verlet substeps kick(h/2) drift(h) kick(h/2):
 *    kick  -- momentum updates from dV_r/du and dW/dtheta, plus the t and phi quadratures
 *    drift -- free radial drift u += h p_u and the exact great-circle polar flow
 *             (mino_polar_flow, which handles the h^2/sin^2theta barrier near the polar axis)
 *  Composition weights: order 2 = {1}; order 4 = Yoshida {w1, w0, w1}; order 6 = Yoshida
 *  solution A {w3, w2, w1, w0, w1, w2, w3} (Yoshida 1990, Phys. Lett. A 150, 262).
 *
 *  This header is shared by Raytracer::propagate_symplectic_impl() (raytracer.cpp), which propagates
 *  single rays to completion, and by applications that need to drive the step themselves, e.g. the
 *  caustic mapper (src/caustic/caustic_3d.cpp), which advances a bundle of rays in lockstep.
 */

#ifndef MINO_STEPPER_H_
#define MINO_STEPPER_H_

#include <cmath>
#include "../include/kerr.h"

// horizon threshold in the radial coordinate u (r = 1 + b cosh u, so u = 0 is the horizon and u = 0.05 is
// r - r+ ~ 1e-3 b).  The radial potential is singular at u = 0 and photon turning points never lie below
// u ~ 0.5 (the circular photon orbit) for any spin, so a ray reaching u < SYMP_U_HORIZON is unambiguously
// captured.
#ifndef SYMP_U_HORIZON
#define SYMP_U_HORIZON 0.05
#endif

// canonical Mino-time state of a ray
template <typename T>
struct MinoState
{
    T u, pu, theta, ptheta, t, phi;
};

template <typename T>
class MinoStepper
{
public:
    T a;            // black hole spin (sign as used for the propagation, i.e. negated for backward tracing)
    T k, h;         // constants of motion (energy, z angular momentum)
    int order;      // composition order: 2 (Verlet), 4 or 6 (Yoshida)
    bool hit_horizon;   // set by step() if the ray reached u <= SYMP_U_HORIZON during the step

    MinoStepper(T a, T k, T h, int order = 6) : a(a), k(k), h(h), hit_horizon(false)
    {
        set_order(order);
    }

    void set_order(int o) { order = (o == 2 || o == 4) ? o : 6; }

    // composition weights for the requested order
    static void weights(int order, const T*& w, int& nsub)
    {
        static constexpr T y4_c  = T(1.2599210498948732);                 // 2^(1/3)
        static constexpr T y4_w1 = T(1) / (2 - y4_c);
        static constexpr T y4_w0 = -y4_c / (2 - y4_c);
        static constexpr T y6_w1 = T(-1.17767998417887);
        static constexpr T y6_w2 = T(0.235573213359357);
        static constexpr T y6_w3 = T(0.784513610477560);
        static constexpr T y6_w0 = T(1) - 2*(y6_w1 + y6_w2 + y6_w3);
        static constexpr T weights2[1] = { T(1) };
        static constexpr T weights4[3] = { y4_w1, y4_w0, y4_w1 };
        static constexpr T weights6[7] = { y6_w3, y6_w2, y6_w1, y6_w0, y6_w1, y6_w2, y6_w3 };
        w    = (order == 2) ? weights2 : (order == 6) ? weights6 : weights4;
        nsub = (order == 2) ? 1        : (order == 6) ? 7        : 3;
    }

    // one Stormer-Verlet substep of Mino time hs
    inline void verlet(MinoState<T>& q, T hs)
    {
        T rr, sd, s2;
        // kick(hs/2)
        mino_r_from_u<T>(q.u, a, rr, sd);
        s2 = sin(q.theta)*sin(q.theta);
        q.pu     -= (hs/2) * mino_dVr_dr<T>(rr, k, h, a) * sd;
        q.ptheta -= (hs/2) * mino_dW_dtheta<T>(q.theta, k, a);
        q.t      += (hs/2) * mino_tdot<T>(rr, q.theta, k, h, a);
        q.phi    += (hs/2) * (mino_phidot<T>(rr, q.theta, k, h, a) - h/s2);
        // drift(hs)
        q.u += hs * q.pu;
        if (q.u <= T(SYMP_U_HORIZON))
        {
            // reached the horizon: stop here (the potential is singular at u = 0, and a kick
            // taken this close to it -- in particular the backward substeps of the Yoshida
            // compositions -- would be unphysically large)
            if (q.u < 0) q.u = 0;
            hit_horizon = true;
            return;
        }
        mino_polar_flow<T>(q.theta, q.ptheta, q.phi, h, hs);
        // kick(hs/2)
        mino_r_from_u<T>(q.u, a, rr, sd);
        s2 = sin(q.theta)*sin(q.theta);
        q.pu     -= (hs/2) * mino_dVr_dr<T>(rr, k, h, a) * sd;
        q.ptheta -= (hs/2) * mino_dW_dtheta<T>(q.theta, k, a);
        q.t      += (hs/2) * mino_tdot<T>(rr, q.theta, k, h, a);
        q.phi    += (hs/2) * (mino_phidot<T>(rr, q.theta, k, h, a) - h/s2);
    }

    // one full composed step of Mino time hstep (resets and may set hit_horizon)
    inline void step(MinoState<T>& q, T hstep)
    {
        const T* w;
        int nsub;
        weights(order, w, nsub);
        hit_horizon = false;
        for (int i = 0; i < nsub && !hit_horizon; i++) verlet(q, w[i]*hstep);
    }
};

//
// Step-size policy shared by the propagator and the bundle tracer: a fixed Mino step h0 in the
// strong field (r <= r_cap), capped outside r_cap by the coordinate-time step max_tstep / |dt/dlambda_M|
// (a constant affine step at large r, growing in proportion to r beyond maxtstep_rlim so that the radial
// step becomes a fixed fraction of r) and by the phi step max_phistep / |dphi/dlambda_M|.
// Non-positive max_tstep / max_phistep / maxtstep_rlim disable the corresponding cap or growth.
// tdot and phidot (dt/dlambda_M, dphi/dlambda_M at the current position) are returned for reuse.
//
template <typename T>
inline T mino_step_size(T r, T theta, T k, T h, T a, T h0, T r_cap,
                        T max_tstep, T maxtstep_rlim, T max_phistep, T& tdot, T& phidot)
{
    tdot   = mino_tdot<T>(r, theta, k, h, a);
    phidot = mino_phidot<T>(r, theta, k, h, a);

    T step = h0;
    if (r > r_cap)
    {
        T tcap = max_tstep;
        if (maxtstep_rlim > 0 && r > maxtstep_rlim) tcap *= r / maxtstep_rlim;
        if (max_tstep > 0 && step > std::abs(tcap / tdot))
            step = std::abs(tcap / tdot);
        if (max_phistep > 0 && step > std::abs(max_phistep / phidot))
            step = std::abs(max_phistep / phidot);
    }
    return step;
}

#endif /* MINO_STEPPER_H_ */
