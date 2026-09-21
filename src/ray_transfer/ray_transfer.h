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
 *  The material carrying the line (velocity field + density) is abstracted as RTField and its concrete
 *  wind models (SphericalBetaWind, ConicalBetaWind), split out into rtfield.h (header-only) so new
 *  wind/material geometries can be added there without this file growing with them; the continuum-sourcing
 *  region is similarly abstracted as ContinuumSource and its concrete geometries (SphericalContinuumSource),
 *  split out into continuum_source.h for the same reason; the line itself is LineTransition (rest energy +
 *  local Doppler width + opacity normalisation); the observer-frame spectral bins are SpectrumGrid.
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
#include <memory>
#include <vector>

#include "../include/kerr.h"
#include "../include/array.h"
#include "../raytracer/raytracer.h"
#include "../raytracer/imageplane.h"
#include "../raytracer/log_imageplane.h"
#include "../raytracer/mino_stepper.h"
#include "../raytracer/ray_destination.h"
#include "rtfield.h"
#include "continuum_source.h"

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
// RayTransferGrid: the pixel grid RayTransfer::run_raytrace() (and any application driving its own pixel
// loop, e.g. ray_transfer_kerr_vs_flat.cpp) iterates over -- abstracted so a uniform (LinearRayTransferGrid)
// or logarithmically-spaced (LogRayTransferGrid, backed by LogImagePlane, raytracer/log_imageplane.h) grid
// can be swapped in without touching RayTransfer's own tracing logic. Mirrors the existing small pluggable-
// interface pattern in this part of the codebase (RayDestination, ray_destination.h; ContinuumSource,
// continuum_source.h; RTField, rtfield.h) -- the core Raytracer/ImagePlane classes use no virtuals at all,
// but their "pluggable behaviour" extension points already do.
// -----------------------------------------------------------------------------------------------
template <typename T>
class RayTransferGrid
{
public:
    virtual ~RayTransferGrid() = default;
    virtual int Nx() const = 0;
    virtual int Ny() const = 0;
    virtual T x(int ix) const = 0;    // pixel-center x coordinate of column ix
    virtual T y(int iy) const = 0;    // pixel-center y coordinate of row iy
    virtual T dx(int ix) const = 0;   // column ix's physical width -- for self-describing output only
    virtual T dy(int iy) const = 0;   // row iy's physical width    -- for self-describing output only
    // Multiplicative factor applied to a pixel's contribution when summing into an aggregate spectrum
    // (RayTransfer::run_raytrace's spec_total/spec_line/continuum_total, and ray_transfer_kerr_vs_flat.cpp's
    // own hand-rolled loop). For the linear grid this is fixed at 1 -- preserving the existing, longstanding
    // *unweighted* summation convention bit-for-bit, not dx(ix)*dy(iy) -- since redefining it as a true area
    // would silently rescale every existing linear-grid run's absolute spec_total/continuum_total numbers.
    // For the log grid it genuinely is the pixel's area dx(ix)*dy(iy), since an unweighted sum over
    // unequal-area pixels would not be physically meaningful (this was the whole point of
    // LogImagePlane::ray_weight()).
    virtual T weight(int ix, int iy) const = 0;
};

// Uniform grid, pixel centres x0+(ix+0.5)*dx, y0+(iy+0.5)*dy -- exactly RayTransfer's own grid formula
// before this class existed (docs/plan_ray_transfer.md Sec 5.24-5.27). weight() == 1 always: see the base
// class comment above for why this is not dx*dy.
template <typename T>
class LinearRayTransferGrid : public RayTransferGrid<T>
{
    int m_Nx, m_Ny;
    T m_x0, m_dx, m_y0, m_dy;
public:
    LinearRayTransferGrid(int Nx, int Ny, T x0, T dx, T y0, T dy)
        : m_Nx(Nx), m_Ny(Ny), m_x0(x0), m_dx(dx), m_y0(y0), m_dy(dy) {}
    int Nx() const override { return m_Nx; }
    int Ny() const override { return m_Ny; }
    T x(int ix) const override { return m_x0 + (ix + T(0.5)) * m_dx; }
    T y(int iy) const override { return m_y0 + (iy + T(0.5)) * m_dy; }
    T dx(int) const override { return m_dx; }
    T dy(int) const override { return m_dy; }
    T weight(int, int) const override { return T(1); }
};

// Logarithmically-spaced grid, backed by a LogImagePlane the caller constructs and owns (the same object
// should also be passed as RayTransfer's `plane` argument, upcast to `const ImagePlane<T>&`, so init_ray's
// physics and this grid's geometry are guaranteed consistent). ray_x/ray_y/ray_weight (log_imageplane.h)
// take a flat ray index i*get_Ny()+j; the ix*Ny()+iy conversion here reproduces that from (ix, iy).
template <typename T>
class LogRayTransferGrid : public RayTransferGrid<T>
{
    const LogImagePlane<T>& m_plane;
public:
    explicit LogRayTransferGrid(const LogImagePlane<T>& plane) : m_plane(plane) {}
    int Nx() const override { return m_plane.get_Nx(); }
    int Ny() const override { return m_plane.get_Ny(); }
    T x(int ix)  const override { return m_plane.ray_x(ix * m_plane.get_Ny()); }
    T y(int iy)  const override { return m_plane.ray_y(iy); }
    T dx(int ix) const override { return m_plane.dx(ix); }
    T dy(int iy) const override { return m_plane.dy(iy); }
    T weight(int ix, int iy) const override { return m_plane.ray_weight(ix * m_plane.get_Ny() + iy); }
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
// The wind's own line-emission source function S(r) (accumulate_step's "source" parameter) has three modes:
//   Density      S = density_scale * density(r,theta,phi) -- a simple, quick placeholder with no tie to
//                how brightly the corona actually illuminates that point; needs re-tuning (density_scale)
//                if the wind or corona parameters change.
//   Illumination S = corona->illumination(r,theta,phi,spin) -- the corona's specific intensity, diluted
//                geometrically and redshifted out to that point (ContinuumSource::illumination); physically
//                tied to the corona's actual brightness, at the cost of assuming its illuminating light
//                reaches every point along an effectively radial path (see
//                SphericalContinuumSource::illumination, continuum_source.h, for the exact redshift factor
//                used and what is approximated).
//   PowerLaw     S = powerlaw_norm * (R_cyl/powerlaw_ref_r)^-powerlaw_index, R_cyl = r*sin(theta) -- a
//                direct function of the wind point's own cylindrical radius, set by
//                set_powerlaw_source() below; independent of any ContinuumSource (works with corona ==
//                nullptr), for extended sources (e.g. DiscContinuumSource, continuum_source.h) where an
//                accurate illumination() would need integrating over the whole source.
// Illumination requires corona != nullptr; falls back to Density (density_scale = 1) if corona is null.
// -----------------------------------------------------------------------------------------------
enum class WindSourceMode { Density, Illumination, PowerLaw };

template <typename T>
class RayTransfer
{
public:
    // Nx, Ny, x0, dx, y0, dy define the class's own image-plane pixel grid for run_raytrace() (pixel
    // centres x0+(ix+0.5)*dx, y0+(iy+0.5)*dy) -- the original, uniform-grid constructor, unchanged from
    // before LogImagePlane support existed. This is a separate grid from `plane`'s own internal Nx/Ny
    // (ImagePlane keeps those private, uses a different x0/xmax/dx-derived convention, and raytracer.h/.cpp/
    // imageplane.h/.cpp are not touched by this library) -- RayTransfer only ever calls plane.init_ray(x, y)
    // at positions of its own choosing. continuum_map/tau_map/flux_cube/spec_line/spec_total (below) are
    // allocated here too, sized from Nx/Ny and bins.energy.size(). Internally builds and owns a
    // LinearRayTransferGrid (below) -- see the LogImagePlane overload just below for why an internal
    // RayTransferGrid still exists even though it is no longer part of either constructor's public API.
    //
    // The symplectic integrator's own controls (step, order, far-field max_tstep) are not constructor
    // arguments -- as with Raytracer, they default to the values already in use elsewhere in this library
    // (order 6, step 1/PRECISION, far-field cap MAXDT/MAXDPHI beyond MAXDT_RLIM) unless overridden by
    // set_symplectic_step()/set_symplectic_order()/set_max_tstep() below, called before trace_pixel()/
    // run_raytrace() (see emissivity.cpp for the same pattern on Raytracer).
    RayTransfer(const ImagePlane<T>& plane, T spin, const RTField<T>& field,
                const LineTransition<T>& line, const SpectrumGrid<T>& bins,
                int Nx, int Ny, T x0, T dx, T y0, T dy,
                const RayDestination<T>* disc = nullptr, const ContinuumSource<T>* corona = nullptr,
                WindSourceMode source_mode = WindSourceMode::Density, T density_scale = 1);

    // Logarithmically-spaced grid: takes only a LogImagePlane, no separate grid parameters, since a
    // LogImagePlane already knows its own grid (get_Nx()/get_Ny()/ray_x()/ray_y()/ray_weight()) -- passing
    // it once here plays both roles a plain ImagePlane+grid pair would otherwise need (the geodesic physics
    // via the inherited ImagePlane::init_ray/get_dist/emit_energy/ray_redshift, and the pixel grid), so the
    // two can never disagree about which object defines the image plane (a real risk with two independent
    // parameters -- e.g. an ImagePlane paired with a grid built from an unrelated LogImagePlane -- that
    // this overload eliminates by construction rather than by convention). Internally builds and owns a
    // LogRayTransferGrid (below): RayTransferGrid still exists as an internal adapter, unifying ImagePlane's
    // pixel-center/(ix,iy) grid convention (which ImagePlane itself has no public API for at all) and
    // LogImagePlane's flat-ray-index one into the single shape run_raytrace() drives polymorphically,
    // without needing two different implementations of run_raytrace() itself.
    RayTransfer(const LogImagePlane<T>& plane, T spin, const RTField<T>& field,
                const LineTransition<T>& line, const SpectrumGrid<T>& bins,
                const RayDestination<T>* disc = nullptr, const ContinuumSource<T>* corona = nullptr,
                WindSourceMode source_mode = WindSourceMode::Density, T density_scale = 1);

    // Symplectic integrator controls -- same meaning and defaults as Raytracer::set_symplectic_step()/
    // set_symplectic_order()/set_max_tstep() (raytracer.h). A wind whose structure varies on scales
    // smaller than the default far-field step (e.g. a steep density gradient close to its own launch
    // radius) needs max_tstep tightened -- see docs/plan_ray_transfer.md Sec 6.
    void set_symplectic_step(T h0) { m_step = h0; }
    T    get_symplectic_step() const { return m_step; }
    void set_symplectic_order(int order) { m_order = (order == 2 || order == 4) ? order : 6; }
    int  get_symplectic_order() const { return m_order; }
    void set_max_tstep(T max, T rlim = MAXDT_RLIM) { m_max_tstep = max; m_maxtstep_rlim = rlim; }

    // WindSourceMode::PowerLaw's normalisation/reference radius/exponent (see the enum comment above).
    // Only read when source_mode == PowerLaw; no effect otherwise.
    void set_powerlaw_source(T norm, T ref_r, T index)
    {
        m_powerlaw_norm = norm; m_powerlaw_ref_r = ref_r; m_powerlaw_index = index;
    }

    // Trace the ray landing at image-plane position (x, y).  Returns true iff the ray terminated on the
    // corona (continuum > 0 in that case; 0 otherwise -- blocked by the disc, escaped, stuck at the step
    // limit, or non-finite initial data, e.g. the on-axis image-plane ray).  The observed flux per bin is
    // continuum * exp(-absorption[j]) + line_emission[j].
    //
    // r_max/steplim mirror Raytracer::run_raytrace()'s own rlim/steplim, passed through to propagate*()
    // rather than stored on the class (raytracer.h/.cpp) -- non-positive (the default) resolves r_max to
    // 1.1 * plane.get_dist() (same convention as CausticBundle::Params::from_plane; a fixed default of,
    // say, 1000 would put the escape radius at or inside the image plane itself for any realistic (larger)
    // observer distance, so rays would "escape" before taking a single step) and steplim to SYMP_STEPLIM.
    bool trace_pixel(T x, T y, std::vector<T>& line_emission, std::vector<T>& absorption, T& continuum,
                      T r_max = -1, int steplim = -1) const;

    // Traces every pixel of the Nx x Ny grid set up by the constructor, filling continuum_map/tau_map/
    // flux_cube/spec_line/spec_total/continuum_total/n_corona/tau_corona_max below in place -- the
    // class-level equivalent of the pixel loop every RayTransfer application previously wrote out for
    // itself (mirrors Raytracer::run_raytrace, which likewise owns and fills its own `rays` array). Pixels
    // are independent (trace_pixel is const, touches no shared state), so the row loop is parallelised
    // internally exactly as it was at the application level. r_max/steplim: see trace_pixel() above --
    // resolved once here (not once per pixel) and passed through to every trace_pixel() call.
    //
    // show_progress: same convention as Raytracer::run_raytrace's own ProgressBar (progress_bar.h) --
    // positive draws a live bar, negative prints a plain "done/total" line, 0 disables it; the magnitude
    // is the update interval in pixels traced.
    void run_raytrace(T r_max = -1, int steplim = -1, int show_progress = 1);

    // Per-pixel/per-bin results, allocated by the constructor and filled by run_raytrace().
    //   continuum_map, tau_map   [ix][iy]: unabsorbed corona continuum, and peak wind optical depth over
    //                             the energy grid (max_j absorption[j]) for that line of sight.
    //   flux_cube                [j][iy][ix] (fits_output.h's write_me_data_cube axis convention):
    //                             continuum*exp(-absorption) + line_emission per energy bin.
    //   spec_line, spec_total    [j]: line emission, and total flux, summed over every pixel.
    //   continuum_total          sum of continuum_map over every pixel.
    //   n_corona                 number of pixels whose ray terminated on the corona.
    //   tau_corona_max           peak tau_map restricted to those pixels -- the absorption trough's actual
    //                            depth against the continuum, as distinct from tau_map's global peak
    //                            (docs/plan_ray_transfer.md Sec 5.19).
    std::unique_ptr<Array2D<T>> continuum_map, tau_map;
    std::unique_ptr<Array3D<T>> flux_cube;
    std::vector<T> spec_line, spec_total;
    T continuum_total = 0;
    T tau_corona_max = 0;
    long n_corona = 0;

    int get_Nx() const { return m_grid->Nx(); }
    int get_Ny() const { return m_grid->Ny(); }

    // Pixel-center coordinates and cell widths, valid regardless of grid type -- for callers that need to
    // record where each pixel actually is (e.g. self-describing FITS output), without needing to know or
    // reconstruct which concrete RayTransferGrid this object built internally.
    T grid_x(int ix) const { return m_grid->x(ix); }
    T grid_y(int iy) const { return m_grid->y(iy); }
    T grid_dx(int ix) const { return m_grid->dx(ix); }
    T grid_dy(int iy) const { return m_grid->dy(iy); }
    // The same per-pixel area-weighting factor run_raytrace() applies internally (see RayTransferGrid's
    // weight() comment) -- for callers driving their own pixel loop over trace_pixel() directly instead of
    // run_raytrace() (e.g. ray_transfer_kerr_vs_flat.cpp).
    T grid_weight(int ix, int iy) const { return m_grid->weight(ix, iy); }

private:
    // Shared tail of both constructors: allocates continuum_map/tau_map/flux_cube/spec_line/spec_total
    // from m_grid->Nx()/Ny(), once m_grid has been built by whichever constructor ran.
    void init_arrays();

    const ImagePlane<T>& m_plane;
    T m_spin;
    const RTField<T>& m_field;
    LineTransition<T> m_line;
    SpectrumGrid<T> m_bins;
    std::unique_ptr<RayTransferGrid<T>> m_grid;   // owned; built by whichever constructor ran (safe: unlike
                                                   // ImagePlane, RayTransferGrid has a virtual destructor)
    const RayDestination<T>* m_disc;
    const ContinuumSource<T>* m_corona;
    int m_order;
    T m_step;
    T m_max_tstep, m_max_phistep, m_maxtstep_rlim;
    WindSourceMode m_source_mode;
    T m_density_scale;
    T m_powerlaw_norm = 0, m_powerlaw_ref_r = 1, m_powerlaw_index = 0;
};

#endif /* RAY_TRANSFER_H_ */
