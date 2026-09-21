/*
 * continuum_source.h
 *
 *  ContinuumSource and its concrete geometries -- split out of ray_transfer.h so that new continuum-source
 *  shapes can be added here without the core RayTransfer/FlatRayTransfer machinery growing with them.
 *  Included by ray_transfer.h; applications normally just include "ray_transfer.h" and get this too.
 *  Header-only (all member functions defined in-class), same rationale as rtfield.h: these are templates,
 *  so their definitions have to be visible at every instantiation site anyway.
 */

#ifndef CONTINUUM_SOURCE_H_
#define CONTINUUM_SOURCE_H_

#include <cmath>

#include "../include/kerr.h"

// -----------------------------------------------------------------------------------------------
// Geometric dilution factor of an isotropically-emitting spherical photosphere of radius R_star, at
// radius r (Mihalas 1978): the mean intensity of the illuminating radiation field at r, relative to the
// specific intensity at the photosphere, under the standard single-scattering approximation (the
// illuminating beam itself is assumed optically thin -- no self-shielding).  Used both by
// SphericalContinuumSource::illumination below and by FlatRayTransfer's star (ray_transfer.cpp).
// -----------------------------------------------------------------------------------------------
template <typename T>
inline T dilution_factor(T r, T R_star)
{
    if (r <= R_star) return T(0.5);
    const T x = R_star / r;
    return T(0.5) * (1 - sqrt(1 - x*x));
}

// -----------------------------------------------------------------------------------------------
// ContinuumSource: the compact, continuum-sourcing region around the black hole (e.g. a corona).
// Parallel in spirit to RTField, but for the continuum source rather than the absorbing/emitting wind
// volume -- kept as its own interface (rather than reusing RTField or RayDestination) since neither fits:
// it is not a stopping *surface* (RayDestination) and it has no density/opacity role (RTField), just a
// geometric extent and an intensity.  "Spherical by default": SphericalContinuumSource is the only
// geometry implemented, but contains() is virtual so other shapes can be added later without changing
// RayTransfer.
//
//   contains()      true inside the source.  The 4-argument crossing-aware overload (prev_theta = theta
//                    from the previous step, mirroring RayDestination::reached()) defaults to the
//                    3-argument pointwise test, which is correct for a genuine volume (SphericalContinuumSource)
//                    -- a zero-thickness surface (DiscContinuumSource) overrides it, since a pointwise test
//                    would essentially never land exactly on the surface.
//   four_velocity()  the source's local rest frame, for redshifting its continuum to the observer;
//                    defaults to the static (non-rotating) observer (same construction
//                    SphericalBetaWind::four_velocity uses at zero boost) -- override for other motion.
//   illumination()   the source's contribution to the wind's line source function at (r,theta,phi): its
//                     specific intensity, diluted geometrically and redshifted from the source out to
//                     that point (I_nu/nu^3 invariance, g^-3 -- same invariant that fixes the source's own
//                     continuum boost and the wind's line emission, ray_transfer.cpp). Pure virtual since
//                     the geometric dilution is shape-specific; SphericalContinuumSource implements it.
//   intensity        local rest-frame specific intensity, flat (energy-independent).
// -----------------------------------------------------------------------------------------------
template <typename T>
class ContinuumSource
{
public:
    T intensity;

    explicit ContinuumSource(T intensity) : intensity(intensity) {}
    virtual ~ContinuumSource() = default;
    virtual bool contains(T r, T theta, T phi) const = 0;
    virtual bool contains(T r, T theta, T phi, T prev_theta) const { return contains(r, theta, phi); }
    virtual T illumination(T r, T theta, T phi, T spin) const = 0;

    virtual void four_velocity(T r, T theta, T phi, T spin, T et[4]) const
    {
        // static (V = 0) observer -- same tetrad construction SphericalBetaWind::four_velocity uses at zero
        // boost, kept unboosted here (non-rotating source, per docs/plan_ray_transfer.md)
        T pos[4] = {0, r, theta, phi};
        T e1[4], e2[4], e3[4];
        tetrad<T>(et, e1, e2, e3, pos, T(0), spin);
    }
};

template <typename T>
class SphericalContinuumSource : public ContinuumSource<T>
{
public:
    T R_corona;

    SphericalContinuumSource(T R_corona, T intensity) : ContinuumSource<T>(intensity), R_corona(R_corona) {}
    bool contains(T r, T theta, T phi) const override { return r <= R_corona; }

    T illumination(T r, T theta, T phi, T spin) const override
    {
        // Geometric dilution: the same solid-angle formula FlatRayTransfer's star uses (dilution_factor,
        // above), treating the source as an isotropically-emitting sphere.
        //
        // Redshift from the source's radius out to this point: for a static (V = 0) observer, the locally
        // measured energy of a photon with conserved energy-at-infinity k is E = k/sqrt(g00) exactly, for
        // *any* Kerr spin and *any* connecting null geodesic -- g00*p^t + g03*p^phi = k identically for any
        // Kerr geodesic (k is this codebase's p_t, and p_t = g_{0 nu} p^nu trivially; verified directly
        // against momentum_from_consts + kerr_metric, not just derived -- docs/plan_ray_transfer.md Sec
        // 5.11), so this is independent of the photon's own h/Q and does not depend on which specific
        // (generally bent) path an illuminating photon takes between the two radii; it is the same
        // static-to-static Killing-energy argument the source's own continuum boost and the
        // Schwarzschild-redshift test already rely on. Evaluating the source and the point at the *same*
        // theta is the one approximation here (a radial illuminating path) -- the dilution factor above is
        // the bigger one: unlike this redshift ratio, it is *not* protected by the Killing-energy
        // cancellation (solid angle depends on how a bundle of geodesics spreads, which curvature/lensing
        // does change), so it can miss lensing magnification, additional bent-light images, and
        // self-occultation -- see Sec 5.11 for why fixing that would need a real ray trace from every wind
        // point, not just two metric evaluations.
        T pos_corona[4] = {0, R_corona, theta, phi};
        T pos_point[4] = {0, r, theta, phi};
        T g_corona[4][4], g_point[4][4];
        kerr_metric<T>(g_corona, pos_corona, spin);
        kerr_metric<T>(g_point, pos_point, spin);
        const T g_illum = sqrt(g_point[0][0] / g_corona[0][0]);   // E_static(source) / E_static(point)
        return this->intensity * dilution_factor<T>(r, R_corona) / (g_illum * g_illum * g_illum);
    }
};

// -----------------------------------------------------------------------------------------------
// DiscContinuumSource: the accretion disc surface itself (theta = pi/2), restricted to an annulus
// R_in <= r <= R_out, as a continuum source -- e.g. an illuminated/reprocessing patch of disc, distinct
// from a compact corona. A zero-thickness surface, so contains(r,theta,phi) alone (no prev_theta) is
// always false; it is only ever "entered" via the crossing-aware overload, exactly as
// DiscWithISCODestination::reached() detects a disc crossing (ray_destination.h).
// -----------------------------------------------------------------------------------------------
template <typename T>
class DiscContinuumSource : public ContinuumSource<T>
{
public:
    T R_in, R_out;

    DiscContinuumSource(T R_in, T R_out, T intensity) : ContinuumSource<T>(intensity), R_in(R_in), R_out(R_out) {}

    bool contains(T r, T theta, T phi) const override { return false; }

    bool contains(T r, T theta, T phi, T prev_theta) const override
    {
        if (r < R_in || r > R_out) return false;
        const T tl = M_PI_2;
        return (prev_theta < tl && theta >= tl) || (prev_theta > tl && theta <= tl);
    }

    // Exact equatorial Keplerian orbit -- disc_velocity_vector (kerr.h) already assumes theta = pi/2, which
    // holds here exactly (the source only exists on that plane).
    void four_velocity(T r, T theta, T phi, T spin, T et[4]) const override
    {
        disc_velocity_vector<T>(et, r, spin, +1);
    }

    // Minimal fallback so the class isn't abstract -- not the intended source function for a
    // RayTransfer application using this source (that would normally use WindSourceMode::PowerLaw
    // instead, see ray_transfer.h): treats the whole annulus as an effective ring at its mean radius,
    // same dilution-factor construction as SphericalContinuumSource::illumination.
    T illumination(T r, T theta, T phi, T spin) const override
    {
        const T R_mid = (R_in + R_out) / 2;
        T pos_disc[4] = {0, R_mid, T(M_PI_2), phi};
        T pos_point[4] = {0, r, theta, phi};
        T g_disc[4][4], g_point[4][4];
        kerr_metric<T>(g_disc, pos_disc, spin);
        kerr_metric<T>(g_point, pos_point, spin);
        const T g_illum = sqrt(g_point[0][0] / g_disc[0][0]);
        return this->intensity * dilution_factor<T>(r, R_mid) / (g_illum * g_illum * g_illum);
    }
};

#endif /* CONTINUUM_SOURCE_H_ */
