/*
 * ray_destination.h
 *
 *  Pluggable ray termination criteria and disc velocity fields for the RK4 ray tracer.
 *
 *  Subclass RayDestination<T> and implement:
 *    reached()    — return true to stop propagation at the current position.
 *    step_limit() — return an upper bound on the integrator step size to prevent
 *                   overshooting the boundary surface.  Default: no limit.
 *    velocity()   — return the angular velocity Ω = dφ/dt of disc material at the
 *                   termination point, for use by redshift(dest*). Return -1 to fall
 *                   back to the Keplerian circular-orbit velocity (default).
 */

#ifndef RAY_DESTINATION_H_
#define RAY_DESTINATION_H_

#include <cmath>
#include <limits>

// Abstract base class — subclass this to define a custom ray termination surface
// and an associated velocity field.
//
// reached() is called after every RK4 position update; return true to stop the ray.
//
// step_limit() returns an upper bound on the integrator step size (in affine parameter)
// that prevents the adaptive RK45 integrator from overshooting the boundary surface in
// a single step.  Given the current position (r,θ,φ) and first-stage momenta
// (pr,pθ,pφ), return the step that would just reach the boundary under linear
// extrapolation.  Return std::numeric_limits<T>::max() when no meaningful limit can be
// defined for the current position or geometry (default).
//
// velocity() returns the scalar angular velocity Ω = dφ/dt of material at (r,θ,φ).
// Return -1 for equatorial Keplerian (default). Used by four_velocity() default.
//
// four_velocity() populates et[4] = {ut, ur, uθ, uφ} (contravariant Boyer-Lindquist)
// for the material at (r,θ,φ). The default computes the circular-orbit 4-velocity from
// velocity(), using the Keplerian value if velocity() returns -1. Override to supply a
// fully general velocity field (e.g. warped disc, outflow) — the vector must satisfy
// g_μν et^μ et^ν = +1 (timelike, +--- signature convention used throughout the codebase).
// spin is the black-hole spin parameter a/M.
template <typename T>
class RayDestination {
public:
    virtual ~RayDestination() = default;
    virtual bool reached(T r, T theta, T phi) const = 0;
    // Crossing-aware overload: prev_theta is theta from the previous step.
    // Subclasses can override this to check for an actual crossing of a surface
    // rather than just the current position.  The default delegates to the
    // positional overload (ignoring prev_theta), which is correct for simple
    // one-sided surfaces such as FlatDiscDestination.
    virtual bool reached(T r, T theta, T phi, T prev_theta) const {
        return reached(r, theta, phi);
    }
    virtual T step_limit(T r, T theta, T phi, T pr, T ptheta, T pphi) const {
        return std::numeric_limits<T>::max();
    }
    virtual T velocity(T r, T theta, T phi) const { return -1; }
    virtual void four_velocity(T r, T theta, T phi, T spin, T et[4]) const
    {
        T V = velocity(r, theta, phi);

        const T rhosq   = r*r + (spin*cos(theta))*(spin*cos(theta));
        const T delta   = r*r - 2*r + spin*spin;
        const T sigmasq = (r*r + spin*spin)*(r*r + spin*spin)
                          - spin*spin*delta*sin(theta)*sin(theta);
        const T e2nu    = rhosq * delta / sigmasq;
        const T e2psi   = sigmasq * sin(theta)*sin(theta) / rhosq;
        const T omega   = 2*spin*r / sigmasq;

        if (V == -1) V = 1 / (spin + r * sqrt(r));   // equatorial Keplerian

        const T gamma_factor = 1 / sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu);
        et[0] = gamma_factor / sqrt(e2nu);
        et[1] = 0;
        et[2] = 0;
        et[3] = gamma_factor * V / sqrt(e2nu);
    }
};

// Concrete implementation equivalent to the existing theta_lim behaviour.
//   theta_lim > 0: stop when theta >= theta_lim  (e.g. M_PI/2 for equatorial disc)
//   theta_lim < 0: stop when theta <= |theta_lim| (reverse integration)
//   theta_lim = 0: never stop on theta (equivalent to passing thetalim=0 to propagate_rk4)
template <typename T>
class FlatDiscDestination : public RayDestination<T> {
    T theta_lim;
public:
    explicit FlatDiscDestination(T theta_lim = M_PI_2) : theta_lim(theta_lim) {}
    bool reached(T r, T theta, T phi) const override {
        if (theta_lim > 0) return theta >= theta_lim;
        if (theta_lim < 0) return theta <= -theta_lim;
        return false;
    }
    T step_limit(T r, T theta, T phi, T pr, T ptheta, T pphi) const override {
        if (theta_lim > 0 && ptheta > 0 && theta < theta_lim)
            return (theta_lim - theta) / ptheta;
        if (theta_lim < 0 && ptheta < 0 && theta > -theta_lim)
            return (-theta_lim - theta) / ptheta;
        return std::numeric_limits<T>::max();
    }
};

// Like FlatDiscDestination but only triggers within the annular disc region r_isco <= r <= r_out.
// Rays inside the ISCO pass through (continuing to the horizon).
// Rays beyond r_out pass through — they may orbit and re-enter the annulus from the other side.
//
// The crossing-aware overload requires prev_theta (theta from the previous propagator step)
// to correctly detect disc crossings regardless of approach direction.  A ray is stopped only
// when theta has actually crossed theta_lim since the last step, catching both "from above"
// (prev < theta_lim, current >= theta_lim) and "from below" (prev > theta_lim, current <=
// theta_lim) crossings.  This avoids the ambiguity of thetadot_sign, which cannot distinguish
// "ray has come from the southern hemisphere" from "ray in the northern hemisphere moving
// away from the disc."
template <typename T>
class DiscWithISCODestination : public RayDestination<T> {
    T theta_lim;
    T r_isco;
    T r_out;   // outer disc boundary; <= 0 means no outer limit
public:
    explicit DiscWithISCODestination(T r_isco, T r_out = -1, T theta_lim = M_PI_2)
        : theta_lim(theta_lim), r_isco(r_isco), r_out(r_out) {}
    bool reached(T r, T theta, T phi) const override {
        if (r < r_isco) return false;
        if (r_out > 0 && r > r_out) return false;
        if (theta_lim > 0) return theta >= theta_lim;
        if (theta_lim < 0) return theta <= -theta_lim;
        return false;
    }
    bool reached(T r, T theta, T phi, T prev_theta) const override {
        if (r < r_isco) return false;
        if (r_out > 0 && r > r_out) return false;
        if (theta_lim > 0)
            return (prev_theta < theta_lim && theta >= theta_lim) ||   // crossed from above
                   (prev_theta > theta_lim && theta <= theta_lim);      // crossed from below
        if (theta_lim < 0) {
            const T tl = -theta_lim;
            return (prev_theta > tl && theta <= tl) ||
                   (prev_theta < tl && theta >= tl);
        }
        return false;
    }
    T step_limit(T r, T theta, T phi, T pr, T ptheta, T pphi) const override {
        if (r < r_isco) return std::numeric_limits<T>::max();
        if (r_out > 0 && r > r_out) return std::numeric_limits<T>::max();
        if (theta_lim > 0 && ptheta > 0 && theta < theta_lim)
            return (theta_lim - theta) / ptheta;
        if (theta_lim < 0 && ptheta < 0 && theta > -theta_lim)
            return (-theta_lim - theta) / ptheta;
        return std::numeric_limits<T>::max();
    }
};

// Flat source plane perpendicular to the observer-BH line of sight, located at
// distance z_s behind the BH (on the far side from the observer).
//
// The observer direction unit vector is n = (sin(incl)*cos(phi0),
// sin(incl)*sin(phi0), cos(incl)) in Cartesian coordinates aligned with the BH
// spin axis.  The plane is defined by the equation
//
//   d(r,θ,φ) = r*(sin(θ)*sin(incl)*cos(φ-φ0) + cos(θ)*cos(incl)) = -z_s
//
// reached() returns true when d ≤ -z_s, i.e. the ray has crossed onto the far
// side.  The initial observer position has d ≈ +dist, so the condition never
// fires until the ray has passed through/around the BH.
//
// source_coords() converts the ray's Boyer-Lindquist position to Cartesian
// (x_s, y_s) on the plane, using the same East/North orientation as the image
// plane:
//   x_s (East):  e1 = (-sin(phi0),           cos(phi0),         0         )
//   y_s (North): e2 = (-cos(incl)*cos(phi0), -cos(incl)*sin(phi0), sin(incl))
template <typename T>
class FlatPlaneDestination : public RayDestination<T> {
public:
    T incl;   // observer inclination (radians)
    T phi0;   // observer azimuthal angle (radians)
    T z_s;    // source plane distance behind BH (positive, gravitational radii)

    FlatPlaneDestination(T incl, T phi0, T z_s)
        : incl(incl), phi0(phi0), z_s(z_s) {}

    // Signed projection of the position vector along the observer direction.
    // Positive = observer side; negative = far side.
    T projection(T r, T theta, T phi) const {
        return r * (sin(theta)*sin(incl)*cos(phi - phi0) + cos(theta)*cos(incl));
    }

    bool reached(T r, T theta, T phi) const override {
        return projection(r, theta, phi) <= -z_s;
    }

    // Convert a Boyer-Lindquist position to source-plane Cartesian coordinates.
    // Use rd.phi (accumulated, not range_phi()-wrapped) — cos/sin are periodic
    // so the result is equivalent to using the normalised phi.
    void source_coords(T r, T theta, T phi, T& x_s, T& y_s) const {
        const T X = r * sin(theta) * cos(phi);
        const T Y = r * sin(theta) * sin(phi);
        const T Z = r * cos(theta);
        // East: e1 = (-sin(phi0), cos(phi0), 0)
        x_s = -X * sin(phi0) + Y * cos(phi0);
        // North: e2 = (-cos(incl)*cos(phi0), -cos(incl)*sin(phi0), sin(incl))
        y_s = -X * cos(incl)*cos(phi0) - Y * cos(incl)*sin(phi0) + Z * sin(incl);
    }
};

#endif /* RAY_DESTINATION_H_ */
