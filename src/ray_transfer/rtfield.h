/*
 * rtfield.h
 *
 *  RTField and its concrete wind models -- split out of ray_transfer.h so that new wind/material
 *  geometries can be added here without the core RayTransfer/FlatRayTransfer machinery growing with them.
 *  Included by ray_transfer.h; applications normally just include "ray_transfer.h" and get this too.
 *  Header-only (all member functions defined in-class): these are templates, so their definitions have to
 *  be visible at every instantiation site anyway, and there's little enough code here that a separate
 *  rtfield.cpp only added indirection without saving anything.
 */

#ifndef RTFIELD_H_
#define RTFIELD_H_

#include <cmath>

#include "../include/kerr.h"

// -----------------------------------------------------------------------------------------------
// RTField: the emitting/absorbing material -- 4-velocity and density at a point.  Deliberately not a
// subclass of RayDestination<T> (ray_destination.h), which describes a stopping *surface*; this
// describes a volume.
//
//   four_velocity()      contravariant Boyer-Lindquist 4-velocity et[4] = (u^t,u^r,u^theta,u^phi),
//                         g_mn et^m et^n = +1, for the Kerr ray-path backend (RayTransfer).
//   flat_four_velocity() contravariant Minkowski 4-velocity et[4] = (u^t,u^x,u^y,u^z) at Cartesian
//                         position (x,y,z), for the flat ray-path backend (FlatRayTransfer).  This is a
//                         genuinely different construction from four_velocity() at spin = 0 (Schwarzschild,
//                         not flat -- see ray_transfer.h's file header), not a redundant special case, even
//                         though both describe the same underlying physical velocity law.
//   density()            local particle density (or an equivalent opacity-scale quantity).
//   in_wind()             false where the field has no material (e.g. inside the stellar photosphere).
// -----------------------------------------------------------------------------------------------
template <typename T>
class RTField
{
public:
    virtual ~RTField() = default;
    virtual void four_velocity(T r, T theta, T phi, T spin, T et[4]) const = 0;
    virtual void flat_four_velocity(T x, T y, T z, T et[4]) const = 0;
    virtual T density(T r, T theta, T phi) const = 0;
    virtual bool in_wind(T r, T theta, T phi) const { return true; }
};

// -----------------------------------------------------------------------------------------------
// SphericalBetaWind: classic beta-velocity-law spherical wind, v(r) = v0 + (v_inf - v0) * (1 - R0/r)^beta,
// launched from a photosphere of radius R0 (also the inner edge of the wind) out to an outer radius
// R_out (pass a large value for an effectively unbounded wind, as the P-Cygni test case does); density
// from mass continuity, n(r) v(r) r^2 = const, normalised so that density(2*R0) == n0.
//
// v0 (> 0, required) is the wind's base velocity at r = R0 -- e.g. a thermal/turbulent speed -- and must
// not be zero: density ~ 1/v(r), and for beta = 1 the column density picked up while v(r) is anywhere
// near v0 is a *logarithmically divergent* integral (its value is independent of how small a numerical
// floor on v(r) is set to, so no amount of floor-tuning avoids it -- confirmed empirically: tightening a
// floor made a spurious spectral spike worse, not better; see docs/plan_ray_transfer.md). Earlier versions
// of this class floored (1 - R0/r) directly instead of giving the wind a real v0, which does not fix this
// (the floored region is exactly this same divergent shell, just relocated) -- v0 removes the divergence
// itself, rather than bounding its symptom, since v(r) is then smooth and bounded away from zero
// everywhere, including at r = R0 exactly.
// -----------------------------------------------------------------------------------------------
template <typename T>
class SphericalBetaWind : public RTField<T>
{
public:
    T v_inf;      // terminal velocity (units of c)
    T v0;         // base velocity at r = R0 (units of c); must be > 0
    T beta_exp;   // velocity-law exponent
    T R0;         // wind base / photosphere radius
    T R_out;      // outer wind radius
    T n0;         // density normalisation: density(2*R0) == n0

    SphericalBetaWind(T v_inf, T v0, T beta_exp, T R0, T R_out, T n0)
        : v_inf(v_inf), v0(v0), beta_exp(beta_exp), R0(R0), R_out(R_out), n0(n0)
    {
    }

    T velocity(T r) const
    {
        if (r <= R0) return v0;
        return v0 + (v_inf - v0) * pow(1 - R0/r, beta_exp);
    }

    T density(T r, T theta, T phi) const override
    {
        if (r <= R0 || r >= R_out) return 0;
        const T v = velocity(r);
        if (v <= 0) return 0;   // v0 == 0 (and hence v(r) == 0 everywhere): no wind material at all
        // mass continuity n(r) v(r) r^2 = const, normalised at r_ref = 2*R0
        const T r_ref = 2*R0;
        const T v_ref = velocity(r_ref);
        return n0 * (v_ref * r_ref*r_ref) / (v * r*r);
    }

    bool in_wind(T r, T theta, T phi) const override { return r > R0 && r < R_out; }

    void four_velocity(T r, T theta, T phi, T spin, T et[4]) const override
    {
        // static (V = 0) Kerr tetrad, boosted radially by the wind speed at r (kerr.h: kerr_metric, tetrad)
        const T beta = velocity(r);
        const T gamma = 1 / sqrt(1 - beta*beta);
        T pos[4] = {0, r, theta, phi};
        T g[4][4], etv[4], e1[4], e2[4], e3[4];
        kerr_metric<T>(g, pos, spin);
        tetrad<T>(etv, e1, e2, e3, pos, T(0), spin);
        for (int mu = 0; mu < 4; mu++)
            et[mu] = gamma*etv[mu] + gamma*beta*e3[mu];
    }

    void flat_four_velocity(T x, T y, T z, T et[4]) const override
    {
        const T r = sqrt(x*x + y*y + z*z);
        const T beta = velocity(r);
        const T gamma = 1 / sqrt(1 - beta*beta);
        et[0] = gamma;
        if (r > 0)
        {
            et[1] = gamma*beta*x/r;
            et[2] = gamma*beta*y/r;
            et[3] = gamma*beta*z/r;
        }
        else
        {
            et[1] = et[2] = et[3] = 0;
        }
    }
};

// DiscLaunched: wind fills theta_lim < theta < pi - theta_lim (hollow near both poles, symmetric about
//               the equator) -- launched close to the disc surface, absent near the polar axis.
// PolarCollimated: wind fills theta < theta_lim || theta > pi - theta_lim (filled cones at both poles,
//               symmetric about the equator) -- collimated along the polar axis, absent near the equator.
// theta_lim is measured from the pole (theta = 0), matching Boyer-Lindquist convention (theta = pi/2 is
// the equatorial plane) used throughout kerr.h/raytracer.h; must be in (0, pi/2).
enum class WindLatitudeMode { DiscLaunched, PolarCollimated };

// -----------------------------------------------------------------------------------------------
// ConicalBetaWind: SphericalBetaWind's beta-velocity-law/density profile (delegated to an internal
// SphericalBetaWind, unchanged -- the radial kinematics/density here do not depend on theta at all, only
// whether a given theta is included), restricted to a range of polar angle theta (WindLatitudeMode
// above) rather than filling the whole sphere. Composition over inheritance/duplication: this is exactly
// SphericalBetaWind's physics with one extra gate in in_wind(), not a different velocity/density law.
// -----------------------------------------------------------------------------------------------
template <typename T>
class ConicalBetaWind : public RTField<T>
{
public:
    SphericalBetaWind<T> radial;
    T theta_lim;
    WindLatitudeMode mode;

    ConicalBetaWind(T v_inf, T v0, T beta_exp, T R0, T R_out, T n0, T theta_lim, WindLatitudeMode mode)
        : radial(v_inf, v0, beta_exp, R0, R_out, n0), theta_lim(theta_lim), mode(mode)
    {
    }

    bool in_theta_range(T theta) const
    {
        const bool near_pole = (theta < theta_lim) || (theta > M_PI - theta_lim);
        return (mode == WindLatitudeMode::PolarCollimated) ? near_pole : !near_pole;
    }

    T density(T r, T theta, T phi) const override { return radial.density(r, theta, phi); }
    bool in_wind(T r, T theta, T phi) const override
    {
        return radial.in_wind(r, theta, phi) && in_theta_range(theta);
    }
    void four_velocity(T r, T theta, T phi, T spin, T et[4]) const override
    {
        radial.four_velocity(r, theta, phi, spin, et);
    }
    void flat_four_velocity(T x, T y, T z, T et[4]) const override
    {
        radial.flat_four_velocity(x, y, z, et);
    }
};

// -----------------------------------------------------------------------------------------------
// DiscSurface: a cylindrical region of gas sitting above the surface of the accretion disc (assumed to
// lie in the equatorial plane, theta = pi/2), between cylindrical radii R_in/R_out and cylindrical
// heights z_min/z_max. Velocity is purely azimuthal, co-rotating with the disc element at the same
// cylindrical radius directly below (Keplerian, disc_velocity() in kerr.h) -- no radial or vertical
// motion. Written symmetric in |z| (both faces of the disc), which is harmless in practice: an
// accompanying opaque disc (e.g. DiscWithISCODestination) always stops a ray at theta = pi/2 before it
// could ever reach the far side, so only the near face is actually traced.
//
// density_index (RadiusPowerLaw/HeightPowerLaw only) is the exponent p in the profiles below; unused
// (may be left at its default) for Constant.
// -----------------------------------------------------------------------------------------------
enum class DiscSurfaceDensityMode { Constant, RadiusPowerLaw, HeightPowerLaw };

template <typename T>
class DiscSurface : public RTField<T>
{
public:
    T R_in, R_out;      // cylindrical radius range
    T z_min, z_max;     // cylindrical height range (above the disc, |z| = |r cos(theta)|)
    DiscSurfaceDensityMode density_mode;
    T n0;               // density normalisation: Constant -> density == n0 throughout;
                         // RadiusPowerLaw -> density(R_in) == n0; HeightPowerLaw -> density(z_min) == n0
    T density_index;    // exponent p (RadiusPowerLaw/HeightPowerLaw only)

    DiscSurface(T R_in, T R_out, T z_min, T z_max, DiscSurfaceDensityMode density_mode, T n0, T density_index = 0)
        : R_in(R_in), R_out(R_out), z_min(z_min), z_max(z_max), density_mode(density_mode),
          n0(n0), density_index(density_index)
    {
    }

    static void cylindrical(T r, T theta, T& R, T& z)
    {
        R = r * sin(theta);
        z = fabs(r * cos(theta));
    }

    bool in_wind(T r, T theta, T phi) const override
    {
        T R, z;
        cylindrical(r, theta, R, z);
        return R >= R_in && R <= R_out && z >= z_min && z <= z_max;
    }

    T density(T r, T theta, T phi) const override
    {
        if (!in_wind(r, theta, phi)) return 0;
        T R, z;
        cylindrical(r, theta, R, z);
        switch (density_mode)
        {
            case DiscSurfaceDensityMode::RadiusPowerLaw: return n0 * pow(R / R_in, -density_index);
            case DiscSurfaceDensityMode::HeightPowerLaw: return n0 * pow(z / z_min, -density_index);
            default: return n0;
        }
    }

    // Purely azimuthal, co-rotating at the Keplerian angular velocity of the disc element at the same
    // cylindrical radius R (disc_velocity(), kerr.h), boosted into the off-equatorial point (r, theta) via
    // the same general locally-rotating-observer construction RayDestination::four_velocity's default uses
    // (ray_destination.h) -- duplicated here (rather than shared) to avoid touching that already-verified
    // file for a handful of lines of metric algebra.
    void four_velocity(T r, T theta, T phi, T spin, T et[4]) const override
    {
        T R, z;
        cylindrical(r, theta, R, z);
        const T V = disc_velocity<T>(R, spin, +1);

        const T rhosq   = r*r + (spin*cos(theta))*(spin*cos(theta));
        const T delta   = r*r - 2*r + spin*spin;
        const T sigmasq = (r*r + spin*spin)*(r*r + spin*spin) - spin*spin*delta*sin(theta)*sin(theta);
        const T e2nu    = rhosq * delta / sigmasq;
        const T e2psi   = sigmasq * sin(theta)*sin(theta) / rhosq;
        const T omega   = 2*spin*r / sigmasq;

        const T gamma_factor = 1 / sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu);
        et[0] = gamma_factor / sqrt(e2nu);
        et[1] = 0;
        et[2] = 0;
        et[3] = gamma_factor * V / sqrt(e2nu);
    }

    // Flat-space equivalent: as if this cylinder were corotating with an accretion disc with Keplerian
    // orbital velocity as a function of (flat, Cartesian) cylindrical radius R = sqrt(x^2+y^2) -- the
    // Newtonian circular-orbit speed v(R) = sqrt(GM/R) = 1/sqrt(R) (GM = 1, kerr.h's convention), which is
    // exactly disc_velocity(R, 0, +1)*R (the spin = 0 coordinate angular velocity times R) -- purely
    // tangential motion in the (x,y) plane, no vertical or radial component.
    void flat_four_velocity(T x, T y, T z, T et[4]) const override
    {
        const T R = sqrt(x*x + y*y);
        if (R <= 0) { et[0] = 1; et[1] = et[2] = et[3] = 0; return; }
        const T omega = disc_velocity<T>(R, T(0), +1);
        const T v = omega * R;
        const T gamma = 1 / sqrt(1 - v*v);
        et[0] = gamma;
        et[1] = -gamma * v * (y / R);
        et[2] =  gamma * v * (x / R);
        et[3] = 0;
    }
};

#endif /* RTFIELD_H_ */
