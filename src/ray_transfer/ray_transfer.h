/*
 * ray_transfer.h
 *
 *  RayTransfer: integrates the radiative transfer equation for a resonance line along traced null
 *  geodesics, accumulating an energy-dependent optical depth (absorption) and an attenuated local
 *  emission contribution at every step of the ray -- rather than stopping at a single destination
 *  surface, as Raytracer/ImagePlane do.
 *
 *  Two ray-path backends share the same per-step accumulation core (accumulate_step):
 *
 *    FlatRayTransfer -- straight-line rays in flat (Minkowski) spacetime, for a spherical wind with no
 *                        central mass (the P-Cygni verification case).  kerr.h fixes the central mass to
 *                        1 geometric unit everywhere, so this is NOT the spin -> 0 limit of the Kerr
 *                        machinery (that is Schwarzschild, with its own horizon and curvature) -- it is a
 *                        genuinely separate, trivial ray path built only from minkowski()/dot_product().
 *
 *    RayTransfer     -- backward-traced rays from an ImagePlane through the Kerr spacetime, using the
 *                        symplectic (Mino-time) stepper (MinoStepper, mino_stepper.h) exactly as
 *                        Raytracer::propagate_symplectic_impl does internally, but driven by RayTransfer's
 *                        own loop so the emission/absorption accumulation can run at every substep --
 *                        mirroring CausticBundle (src/caustic/caustic_bundle.h), which needed the same
 *                        per-step access and for the same reason (propagate_symplectic_impl is protected
 *                        and has no per-step hook).  No code in raytracer.h/raytracer.cpp is touched; only
 *                        their public API is reused (ImagePlane::init_ray, Raytracer::emit_energy,
 *                        Raytracer::ray_redshift).
 *
 *  The material carrying the line (velocity field + density) is abstracted as RTField; the line itself as
 *  LineTransition (rest energy + local Doppler width + opacity normalisation); the observer-frame
 *  spectral bins as SpectrumGrid.
 *
 *  Proper path length: opacity and emissivity are rest-frame quantities of the material, so the length
 *  entering the optical-depth integral is not the metric's ds (identically zero along a null geodesic,
 *  ds^2 = g_mn dx^m dx^n = (k.k) dlambda^2 = 0) but the distance swept through the local fluid element as
 *  measured by an observer comoving with the material.  Writing the photon's null tangent as
 *  k^mu = E_loc (u^mu + n^mu) (E_loc = g_mn u^m k^n, n^mu the local unit propagation direction), projecting
 *  dx^mu/dlambda = k^mu onto the observer's proper-time and proper-distance axes gives
 *  dtau_local/dlambda = dl_local/dlambda = E_loc, i.e. dl_local = E_loc * dlambda -- the same E_loc that
 *  fixes the local (comoving) line resonance.  Since the traced ray's own energy at the observer is fixed
 *  (by construction, independent of the energy bin being evaluated), dl_local is common to every bin and
 *  is computed once per step from the ratio g = E_loc / E_obs of the traced ray (see docs/plan_ray_transfer.md).
 */

#ifndef RAY_TRANSFER_H_
#define RAY_TRANSFER_H_

#include <cmath>
#include <vector>

#include "../include/kerr.h"
#include "../raytracer/raytracer.h"
#include "../raytracer/imageplane.h"
#include "../raytracer/mino_stepper.h"
#include "../raytracer/ray_destination.h"

// -----------------------------------------------------------------------------------------------
// Line transition: rest energy, local (thermal + turbulent) Doppler width, and opacity normalisation.
// profile() is normalised in energy so that integrating it over all delta_E gives 1 (Gaussian).
// -----------------------------------------------------------------------------------------------
template <typename T>
struct LineTransition
{
    T rest_energy;
    T doppler_width;   // Gaussian sigma, same energy units as rest_energy
    T kappa0;          // opacity normalisation: kappa0 * density * profile * dl_local = d_tau

    T profile(T delta_E) const
    {
        const T x = delta_E / doppler_width;
        return exp(T(-0.5) * x * x) / (doppler_width * sqrt(2 * M_PI));
    }
};

// -----------------------------------------------------------------------------------------------
// Fixed observer-frame energy grid for the output spectra.
// -----------------------------------------------------------------------------------------------
template <typename T>
struct SpectrumGrid
{
    std::vector<T> energy;

    static SpectrumGrid<T> linspace(T e_min, T e_max, int n)
    {
        SpectrumGrid<T> g;
        g.energy.resize(n);
        for (int i = 0; i < n; i++)
            g.energy[i] = e_min + (e_max - e_min) * T(i) / T(n - 1);
        return g;
    }
};

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
//                         not flat -- see the file header), not a redundant special case, even though both
//                         describe the same underlying physical velocity law.
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

    SphericalBetaWind(T v_inf, T v0, T beta_exp, T R0, T R_out, T n0);

    T velocity(T r) const;
    T density(T r, T theta, T phi) const override;
    bool in_wind(T r, T theta, T phi) const override { return r > R0 && r < R_out; }
    void four_velocity(T r, T theta, T phi, T spin, T et[4]) const override;
    void flat_four_velocity(T x, T y, T z, T et[4]) const override;
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
// Corona: the compact, continuum-sourcing region around the black hole.  Parallel in spirit to RTField,
// but for the continuum source rather than the absorbing/emitting wind volume -- kept as its own
// interface (rather than reusing RTField or RayDestination) since neither fits: it is not a stopping
// *surface* (RayDestination) and it has no density/opacity role (RTField), just a geometric extent and an
// intensity.  "Spherical by default": SphericalCorona is the only geometry implemented, but contains() is
// virtual so other shapes can be added later without changing RayTransfer.
//
//   contains()      true inside the corona.
//   four_velocity()  the corona's local rest frame, for redshifting its continuum to the observer;
//                    defaults to the static (non-rotating) observer (same construction
//                    SphericalBetaWind::four_velocity uses at zero boost) -- override for other motion.
//   illumination()   the corona's contribution to the wind's line source function at (r,theta,phi): its
//                     specific intensity, diluted geometrically and redshifted from the corona out to
//                     that point (I_nu/nu^3 invariance, g^-3 -- same invariant that fixes the corona's own
//                     continuum boost and the wind's line emission, ray_transfer.cpp). Pure virtual since
//                     the geometric dilution is shape-specific; SphericalCorona implements it.
//   intensity        local rest-frame specific intensity, flat (energy-independent).
// -----------------------------------------------------------------------------------------------
template <typename T>
class Corona
{
public:
    T intensity;

    explicit Corona(T intensity) : intensity(intensity) {}
    virtual ~Corona() = default;
    virtual bool contains(T r, T theta, T phi) const = 0;
    virtual void four_velocity(T r, T theta, T phi, T spin, T et[4]) const;
    virtual T illumination(T r, T theta, T phi, T spin) const = 0;
};

template <typename T>
class SphericalCorona : public Corona<T>
{
public:
    T R_corona;

    SphericalCorona(T R_corona, T intensity) : Corona<T>(intensity), R_corona(R_corona) {}
    bool contains(T r, T theta, T phi) const override { return r <= R_corona; }
    T illumination(T r, T theta, T phi, T spin) const override;
};

// -----------------------------------------------------------------------------------------------
// Geometric dilution factor of an isotropically-emitting spherical photosphere of radius R_star, at
// radius r (Mihalas 1978): the mean intensity of the illuminating radiation field at r, relative to the
// specific intensity at the photosphere, under the standard single-scattering approximation (the
// illuminating beam itself is assumed optically thin -- no self-shielding).
// -----------------------------------------------------------------------------------------------
template <typename T>
inline T dilution_factor(T r, T R_star)
{
    if (r <= R_star) return T(0.5);
    const T x = R_star / r;
    return T(0.5) * (1 - sqrt(1 - x*x));
}

// -----------------------------------------------------------------------------------------------
// Shared per-step accumulation: adds this step's contribution to the absorption (optical depth) and
// emission spectra, for every energy bin.
//
//   d_lambda  increment of the traced ray's own affine parameter over this step.
//   g         ratio E_loc/E_obs of the traced ray at this step (purely geometric: the local, comoving
//             energy of a photon that would be observed at any fixed E_obs, per unit E_obs).  Determines
//             both the local resonance energy for each bin (E_loc_j = bins.energy[j] * g) and the local
//             proper path length dl_local = g * d_lambda (see the file header derivation; dl_local does
//             not depend on which bin's hypothetical E_obs is being evaluated).
//   density   local material density (or opacity-scale quantity) at this point.
//   source    local rest-frame scattering/emission source function S(E_loc) (same units as the local
//             specific intensity, e.g. LineTransition::rest_energy's units).
//   absorption, emission  per-bin accumulators, updated in place; emission uses the optical depth
//             accumulated *before* this step's own contribution, so self-absorption within one step is
//             not double counted, and is weighted by g^-3 (I_nu/nu^3 invariance: emission of local
//             specific intensity ~ source, redshifted by g = E_loc/E_obs, contributes I_obs ~ source * g^-3
//             to the observer -- see ray_transfer.cpp for the full derivation from the invariant transfer
//             equation). The same g^-3 (not g^3) sets the corona's continuum boost in RayTransfer.
// -----------------------------------------------------------------------------------------------
template <typename T>
void accumulate_step(T d_lambda, T g, T density, T source,
                      const LineTransition<T>& line, const SpectrumGrid<T>& bins,
                      std::vector<T>& absorption, std::vector<T>& emission);

// -----------------------------------------------------------------------------------------------
// FlatRayTransfer: straight-line rays through a spherically symmetric wind in flat spacetime, for the
// P-Cygni verification case.  The ray at impact parameter p travels along z at fixed x = p, y = 0, from
// z_obs (near the observer) down to z_far (upstream of the wind); the photon's fixed lab-frame momentum
// is p^mu = (1, 0, 0, 1) (moving towards +z, i.e. towards the observer), so the observer-frame energy is
// always 1 and g is just the photon's local energy in the material's rest frame.
// -----------------------------------------------------------------------------------------------
template <typename T>
class FlatRayTransfer
{
public:
    FlatRayTransfer(const RTField<T>& field, const LineTransition<T>& line, const SpectrumGrid<T>& bins);

    // Trace one ray of impact parameter p from z = z_obs down to z = z_far, in steps of dz.  R_star sets
    // the photosphere from which the (flat, energy-independent) stellar continuum of intensity I_star
    // originates for p < R_star (0 otherwise), and also the illumination used by the dilution factor.
    void trace_ray(T p, T z_obs, T z_far, T dz, T R_star, T I_star,
                    std::vector<T>& emission, std::vector<T>& absorption) const;

private:
    const RTField<T>& m_field;
    LineTransition<T> m_line;
    SpectrumGrid<T> m_bins;
};

// -----------------------------------------------------------------------------------------------
// RayTransfer: backward-traced rays from an ImagePlane through the Kerr spacetime, with the symplectic
// (Mino-time) stepper driven directly (see the file header).  spin is the physical black-hole spin used
// to build `plane` (ImagePlane negates it internally for the backward trace; Raytracer::spin is protected,
// so -- exactly as CausticBundle::Params::from_plane does -- the physical value has to be supplied
// separately here rather than read back off `plane`).
//
// disc (optional, default none): a stopping surface the ray cannot pass through, e.g.
// DiscWithISCODestination (ray_destination.h) for an equatorial accretion disc -- material behind it is
// never seen, so the trace stops there with no continuum, whatever line_emission/absorption was
// accumulated up to that point standing as the final result for that pixel.
//
// corona (optional, default none): the compact continuum-sourcing region.  Only rays that terminate on
// the corona contribute a continuum; the wind's line emission/absorption is accumulated along the whole
// traced path regardless of the ray's eventual fate (corona, disc, escape, step limit).
//
// The wind's own line-emission source function S(r) (accumulate_step's "source" parameter) has two modes:
//   Density      S = density_scale * density(r,theta,phi) -- a simple, quick placeholder with no tie to
//                how brightly the corona actually illuminates that point; needs re-tuning (density_scale)
//                if the wind or corona parameters change.
//   Illumination S = corona->illumination(r,theta,phi,spin) -- the corona's specific intensity, diluted
//                geometrically and redshifted out to that point (Corona::illumination); physically tied
//                to the corona's actual brightness, at the cost of assuming its illuminating light
//                reaches every point along an effectively radial path (see SphericalCorona::illumination,
//                ray_transfer.cpp, for the exact redshift factor used and what is approximated).
// Illumination requires corona != nullptr; falls back to Density (density_scale = 1) if corona is null.
// -----------------------------------------------------------------------------------------------
enum class WindSourceMode { Density, Illumination };

template <typename T>
class RayTransfer
{
public:
    // r_max <= 0 (default) uses 1.1 * plane.get_dist() (same convention as CausticBundle::Params::from_plane)
    // -- a fixed default of, say, 1000 would put the escape radius at or inside the image plane itself for
    // any realistic (larger) observer distance, so rays would "escape" before taking a single step.
    // max_tstep/max_phistep set the far-field step cap beyond r_cap = 2*horizon (mino_step_size,
    // mino_stepper.h), same meaning as Raytracer::set_max_tstep(); default to the same library-wide
    // defaults (MAXDT/MAXDPHI, raytracer.h) Raytracer itself uses when unset.  A wind whose structure
    // varies on scales smaller than the default far-field step (e.g. a steep density gradient close to
    // its own launch radius) needs these tightened -- see docs/plan_ray_transfer.md Sec 6.
    RayTransfer(const ImagePlane<T>& plane, T spin, const RTField<T>& field,
                const LineTransition<T>& line, const SpectrumGrid<T>& bins,
                const RayDestination<T>* disc = nullptr, const Corona<T>* corona = nullptr,
                int symp_order = 6, T symp_step = -1, T r_max = -1, int steplim = SYMP_STEPLIM,
                T max_tstep = MAXDT, T max_phistep = MAXDPHI, T maxtstep_rlim = MAXDT_RLIM,
                WindSourceMode source_mode = WindSourceMode::Density, T density_scale = 1);

    // Trace the ray landing at image-plane position (x, y).  Returns true iff the ray terminated on the
    // corona (continuum > 0 in that case; 0 otherwise -- blocked by the disc, escaped, stuck at the step
    // limit, or non-finite initial data, e.g. the on-axis image-plane ray).  The observed flux per bin is
    // continuum * exp(-absorption[j]) + line_emission[j].
    bool trace_pixel(T x, T y, std::vector<T>& line_emission, std::vector<T>& absorption, T& continuum) const;

private:
    const ImagePlane<T>& m_plane;
    T m_spin;
    const RTField<T>& m_field;
    LineTransition<T> m_line;
    SpectrumGrid<T> m_bins;
    const RayDestination<T>* m_disc;
    const Corona<T>* m_corona;
    int m_order;
    T m_step, m_r_max;
    int m_steplim;
    T m_max_tstep, m_max_phistep, m_maxtstep_rlim;
    WindSourceMode m_source_mode;
    T m_density_scale;
};

#endif /* RAY_TRANSFER_H_ */
