/*
 * caustic_bundle.h
 *
 *  Lockstep ray-bundle tracer used by caustic_3d (src/caustic/caustic_3d.cpp) to locate the caustics of
 *  the family of rays reaching an observer: a centre ray and four offset rays (+/- delta in the image-plane
 *  x and y directions) are advanced with identical Mino-time steps (mino_stepper.h) and the Jacobian
 *
 *      J(tau) = det[ dX/dx , dX/dy , dX/dtau ]
 *
 *  of the ray family is formed by central differences of the Cartesian positions of the offset rays and
 *  the Mino-time velocity of the centre ray.  Sign changes of J along the centre ray are caustic crossings
 *  and are located by bisection on the fraction of the step.  See caustic_3d.cpp for the full description
 *  and the output format; src/tests/caustic_3d_test.cpp checks the tracer in the Schwarzschild limit.
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
#include "../raytracer/mino_stepper.h"
#include "../include/kerr.h"

using namespace std;

// The bundle is stopped once the centre ray reaches u < BUNDLE_U_STOP in the radial coordinate of the
// Mino-time stepper (r = 1 + b cosh u, b = sqrt(1 - a^2); u = 0 is the horizon and u = 0.25 is
// r - r+ = 0.03 b).  Within that distance of the horizon the finite differences between the bundle rays
// are no longer trustworthy (the radial potential is singular at u = 0 and the composed step's backward
// substeps approach it), producing spurious sign changes of J on the last step before capture.
#define BUNDLE_U_STOP 0.25

// pixel termination status codes (STATUS extension)
enum BundleStatus { STATUS_ESCAPED = 0, STATUS_HORIZON = 1, STATUS_STEPLIM = 2, STATUS_SPLIT = 3,
                    STATUS_SKIPPED = 4, STATUS_NAN = 5 };

struct CausticPoint
{
    int ix, iy, n;
    double ximg, yimg;
    double tau, t, r, theta, phi;
    double X, Y, Z;
    int rflips, eqcross, dj_sign;
    double logscale;
};

struct PixelResult
{
    int status;
    int ncaust;
    int eqcross;
    double tau_end;
    double r1, theta1, phi1;
};

// one member of a ray bundle: canonical Mino-time state plus its constants of motion
struct BundleRay
{
    MinoState<double> s;
    double k, h;
};

struct BundleParams
{
    double a;              // spin used for the propagation (negated for backward tracing)
    double horizon;
    double h0;             // Mino-time step in the strong field
    double r_cap;          // far-field step caps apply outside this radius
    double max_tstep, maxtstep_rlim, max_phistep;
    double r_max;          // escape radius
    double delta;          // bundle offset in the image plane (rg)
    double delta_max;      // rescale the bundle when an offset ray is further than this from the centre
    int order;             // composition order of the symplectic stepper
    int steplim;
    int max_caustics;      // maximum number of caustic points stored per ray
};

static inline double wrap_phi(double phi)
{
    return atan2(sin(phi), cos(phi));
}

// Cartesian position of a bundle ray (kerr.h cartesian())
static inline void ray_position(const BundleRay& b, double a, double X[3])
{
    double r, sd;
    mino_r_from_u<double>(b.s.u, a, r, sd);
    cartesian<double>(X[0], X[1], X[2], r, b.s.theta, b.s.phi, a);
}

// Cartesian velocity dX/dlambda_M of a bundle ray
static inline void ray_velocity(const BundleRay& b, double a, double V[3])
{
    double r, sd;
    mino_r_from_u<double>(b.s.u, a, r, sd);
    const double rdot  = sd * b.s.pu;                                        // dr/dlambda_M = b sinh(u) p_u
    const double thdot = b.s.ptheta;
    const double phdot = mino_phidot<double>(r, b.s.theta, b.k, b.h, a);

    const double R  = sqrt(r*r + a*a);
    const double dR = r / R;
    const double st = sin(b.s.theta), ct = cos(b.s.theta);
    const double sp = sin(b.s.phi),   cp = cos(b.s.phi);

    V[0] = dR*rdot*st*cp + R*ct*cp*thdot - R*st*sp*phdot;
    V[1] = dR*rdot*st*sp + R*ct*sp*thdot + R*st*cp*phdot;
    V[2] = rdot*ct - r*st*thdot;
}

static inline double det3(const double u[3], const double v[3], const double w[3])
{
    return u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0]) + u[2]*(v[0]*w[1] - v[1]*w[0]);
}

// Jacobian J = det[dX/dx, dX/dy, dX/dtau] of the ray family from the bundle
// b[0] centre, b[1] +x, b[2] -x, b[3] +y, b[4] -y
static inline double bundle_jacobian(const BundleRay b[5], double a, double delta)
{
    double X[5][3];
    for (int i = 0; i < 5; i++) ray_position(b[i], a, X[i]);
    double xi_x[3], xi_y[3], V[3];
    for (int c = 0; c < 3; c++)
    {
        xi_x[c] = (X[1][c] - X[2][c]) / (2*delta);
        xi_y[c] = (X[3][c] - X[4][c]) / (2*delta);
    }
    ray_velocity(b[0], a, V);
    return det3(xi_x, xi_y, V);
}

// advance all five rays of a bundle by the same Mino-time step; returns the index of the first ray
// that reached the horizon (or -1)
static inline int bundle_step(BundleRay b[5], MinoStepper<double> st[5], double step)
{
    int horizon_ray = -1;
    for (int i = 0; i < 5; i++)
    {
        st[i].h = b[i].h;
        st[i].step(b[i].s, step);
        if (st[i].hit_horizon && horizon_ray < 0) horizon_ray = i;
    }
    return horizon_ray;
}

static inline bool bundle_finite(const BundleRay b[5])
{
    for (int i = 0; i < 5; i++)
        if (!isfinite(b[i].s.u) || !isfinite(b[i].s.theta) || !isfinite(b[i].s.pu) || !isfinite(b[i].s.ptheta)
            || !isfinite(b[i].s.phi))
            return false;
    return true;
}

static inline bool crossed_equator(double theta_prev, double theta)
{
    return (theta_prev < M_PI_2 && theta >= M_PI_2) || (theta_prev > M_PI_2 && theta <= M_PI_2);
}

// rescale the deviation of offset pair (i, j) from the centre ray by s (in the full state and in h)
static inline void rescale_pair(BundleRay b[5], int i, int j, double s)
{
    const BundleRay& c = b[0];
    for (int m : {i, j})
    {
        b[m].s.u      = c.s.u      + s*(b[m].s.u      - c.s.u);
        b[m].s.pu     = c.s.pu     + s*(b[m].s.pu     - c.s.pu);
        b[m].s.theta  = c.s.theta  + s*(b[m].s.theta  - c.s.theta);
        b[m].s.ptheta = c.s.ptheta + s*(b[m].s.ptheta - c.s.ptheta);
        b[m].s.t      = c.s.t      + s*(b[m].s.t      - c.s.t);
        // phi is taken modulo 2pi: offset rays passing a pole on opposite sides of the axis end up at
        // the same azimuth with BL phi values differing by 2pi
        b[m].s.phi    = c.s.phi    + s*wrap_phi(b[m].s.phi - c.s.phi);
        b[m].h        = c.h        + s*(b[m].h        - c.h);
    }
}

//
// Trace one bundle from the image plane until the centre ray falls through the horizon or escapes,
// recording every caustic crossing in out.
//
static void trace_bundle(BundleRay b[5], const BundleParams& P, int ix, int iy, double ximg, double yimg,
                         vector<CausticPoint>& out, PixelResult& res, ostream* dump = nullptr)
{
    MinoStepper<double> st[5] = { MinoStepper<double>(P.a, b[0].k, b[0].h, P.order),
                                  MinoStepper<double>(P.a, b[1].k, b[1].h, P.order),
                                  MinoStepper<double>(P.a, b[2].k, b[2].h, P.order),
                                  MinoStepper<double>(P.a, b[3].k, b[3].h, P.order),
                                  MinoStepper<double>(P.a, b[4].k, b[4].h, P.order) };

    res.status = STATUS_STEPLIM;
    res.ncaust = 0;
    res.eqcross = 0;
    res.tau_end = 0;
    res.r1 = res.theta1 = res.phi1 = 0;

    int rflips = 0, eqcross = 0;
    double logscale = 0;
    double tau = 0;
    int steps = 0;

    double J_prev = bundle_jacobian(b, P.a, P.delta);

    BundleRay prev[5], trial[5];

    double r, sd;
    mino_r_from_u<double>(b[0].s.u, P.a, r, sd);

    // optional per-step dump of the centre ray and the Jacobian (diagnostics)
    auto dump_state = [&](int step_no, double J_now)
    {
        if (dump == nullptr) return;
        double X[3];
        ray_position(b[0], P.a, X);
        (*dump) << step_no << ' ' << setprecision(12) << tau << ' ' << b[0].s.t << ' ' << r << ' ' << b[0].s.theta
                << ' ' << b[0].s.phi << ' ' << X[0] << ' ' << X[1] << ' ' << X[2] << ' ' << J_now << ' '
                << logscale << ' ' << rflips << ' ' << eqcross << '\n';
    };
    dump_state(0, J_prev);

    while (steps < P.steplim)
    {
        ++steps;

        // step size from the centre ray (same policy as Raytracer::propagate_symplectic)
        double tdot, phidot;
        const double step = mino_step_size<double>(r, b[0].s.theta, b[0].k, b[0].h, P.a, P.h0, P.r_cap,
                                                   P.max_tstep, P.maxtstep_rlim, P.max_phistep, tdot, phidot);

        for (int i = 0; i < 5; i++) prev[i] = b[i];
        const int horizon_ray = bundle_step(b, st, step);

        if (horizon_ray == 0 || b[0].s.u < BUNDLE_U_STOP)
        {
            res.status = STATUS_HORIZON;
            break;
        }
        bool offset_captured = (horizon_ray > 0);
        for (int i = 1; i < 5; i++)
            if (b[i].s.u < BUNDLE_U_STOP) offset_captured = true;
        if (offset_captured)
        {
            // an offset ray was captured while the centre ray was not: the bundle straddles the
            // critical curve and the finite differences are no longer meaningful
            res.status = STATUS_SPLIT;
            break;
        }
        if (!bundle_finite(b))
        {
            for (int i = 0; i < 5; i++) b[i] = prev[i];
            res.status = STATUS_NAN;
            break;
        }

        tau += step;
        mino_r_from_u<double>(b[0].s.u, P.a, r, sd);

        // --- caustic crossing: sign change of J across the step ---
        double J = bundle_jacobian(b, P.a, P.delta);
        if (isfinite(J) && isfinite(J_prev) && J_prev * J < 0)
        {
            // bisection on the fraction of the step; the crossing lies in (lo, hi]
            double lo = 0, hi = step;
            for (int it = 0; it < 60 && (hi - lo) > 1e-13*step; it++)
            {
                const double mid = (lo + hi) / 2;
                for (int i = 0; i < 5; i++) trial[i] = prev[i];
                const int hr = bundle_step(trial, st, mid);
                double Jm = (hr < 0) ? bundle_jacobian(trial, P.a, P.delta) : numeric_limits<double>::quiet_NaN();
                if (hr >= 0 || !isfinite(Jm) || Jm * J_prev < 0) hi = mid; else lo = mid;
            }
            const double frac = (lo + hi) / 2;
            for (int i = 0; i < 5; i++) trial[i] = prev[i];
            bundle_step(trial, st, frac);

            double rc, sdc;
            mino_r_from_u<double>(trial[0].s.u, P.a, rc, sdc);

            CausticPoint cp;
            cp.ix = ix; cp.iy = iy;
            cp.ximg = ximg; cp.yimg = yimg;
            cp.n = res.ncaust + 1;
            cp.tau = tau - step + frac;
            cp.t = trial[0].s.t;
            cp.r = rc;
            cp.theta = trial[0].s.theta;
            cp.phi = wrap_phi(trial[0].s.phi);
            cartesian<double>(cp.X, cp.Y, cp.Z, rc, trial[0].s.theta, trial[0].s.phi, P.a);
            cp.rflips  = rflips  + ((prev[0].s.pu * trial[0].s.pu < 0) ? 1 : 0);
            cp.eqcross = eqcross + (crossed_equator(prev[0].s.theta, trial[0].s.theta) ? 1 : 0);
            cp.dj_sign = (J > 0) ? 1 : -1;
            cp.logscale = logscale;

            if (res.ncaust < P.max_caustics) out.push_back(cp);
            if (res.ncaust == 0)
            {
                res.r1 = cp.r; res.theta1 = cp.theta; res.phi1 = cp.phi;
            }
            ++res.ncaust;
        }
        J_prev = J;
        dump_state(steps, J);

        // --- bookkeeping on the centre ray ---
        if (prev[0].s.pu * b[0].s.pu < 0) ++rflips;
        if (crossed_equator(prev[0].s.theta, b[0].s.theta)) ++eqcross;

        if (r >= P.r_max)
        {
            res.status = STATUS_ESCAPED;
            break;
        }

        // --- keep the bundle in the linear regime: rescale each offset pair when it has grown too far ---
        {
            double Xc[3], Xi[3], Xj[3];
            ray_position(b[0], P.a, Xc);
            const int pairs[2][2] = { {1, 2}, {3, 4} };
            for (int p = 0; p < 2; p++)
            {
                const int i = pairs[p][0], j = pairs[p][1];
                ray_position(b[i], P.a, Xi);
                ray_position(b[j], P.a, Xj);
                double sep = 0;
                for (int c = 0; c < 3; c++)
                {
                    sep = max(sep, fabs(Xi[c] - Xc[c]));
                    sep = max(sep, fabs(Xj[c] - Xc[c]));
                }
                if (sep > P.delta_max)
                {
                    const double s = P.delta / sep;
                    rescale_pair(b, i, j, s);
                    logscale += log10(1/s);
                    J_prev *= s;   // keep the finite-difference J continuous across the rescaling
                }
            }
        }
    }

    res.eqcross = eqcross;
    res.tau_end = tau;
}



// Fill a bundle from ray index pix of five ImagePlane-style ray arrays (centre, +x, -x, +y, -y), converting
// the Boyer-Lindquist initial data to the canonical Mino-time state.  Returns false if the initial data of
// any member is undefined (e.g. the x = y = 0 ray, whose constants of motion are NaN in ImagePlane).
template <typename RayArray>
static inline bool bundle_from_rays(const RayArray* const rays[5], int pix, double a, BundleRay b[5])
{
    bool ok = true;
    for (int i = 0; i < 5; i++)
    {
        const Ray<double>& ray = rays[i][pix];
        b[i].k = ray.k;
        b[i].h = ray.h;
        b[i].s.t = ray.t;
        b[i].s.theta = ray.theta;
        b[i].s.phi = ray.phi;
        if (!std::isfinite(ray.h) || !std::isfinite(ray.Q) || !std::isfinite(ray.theta) || !std::isfinite(ray.phi)) ok = false;
        mino_init<double>(ray.r, ray.theta, ray.k, ray.h, ray.Q, ray.rdot_sign, ray.thetadot_sign, a,
                          b[i].s.u, b[i].s.pu, b[i].s.ptheta);
    }
    return ok;
}

#endif /* CAUSTIC_BUNDLE_H_ */
