/*
 * ray_transfer.cpp
 *
 *  See ray_transfer.h for the design.
 */

#include "ray_transfer.h"
#include <algorithm>
#include <iostream>
#include "../include/progress_bar.h"

// =================================================================================================
// accumulate_step
// =================================================================================================

template <typename T>
void accumulate_step(T d_lambda, T g, T density, T source,
                      const LineTransition<T>& line, const SpectrumGrid<T>& bins,
                      std::vector<T>& absorption, std::vector<T>& emission)
{
    const T dl_local = g * d_lambda;
    if (!(dl_local > 0) || density <= 0) return;

    for (size_t j = 0; j < bins.energy.size(); j++)
    {
        const T E_loc = bins.energy[j] * g;
        const T d_tau = density * line.kappa0 * line.profile(E_loc - line.rest_energy) * dl_local;
        // The invariant GR/SR transfer equation, in terms of O = I_nu/nu^3 (conserved along the ray in
        // vacuum) and the local emission/absorption coefficients j(E_loc) = density*kappa0*profile*source,
        // alpha(E_loc) = density*kappa0*profile, is d(O)/dl_local = j(E_loc)/E_loc^3 - alpha(E_loc)*O
        // (using dl_local = E_loc*dlambda to eliminate dlambda from the usual dO/dlambda form). Its formal
        // solution, converted back to the observed specific intensity I_obs = O*E_obs^3, is
        //   I_obs += j(E_loc) * (E_obs/E_loc)^3 * exp(-tau) * dl_local = j(E_loc) * g^-3 * exp(-tau) * dl_local
        // (E_obs/E_loc = 1/g) -- the same invariant that fixes the corona's continuum boost (g^-3, not
        // g^3: redshifted emission, g > 1, must come out *dimmer*, not brighter). emission sees the
        // optical depth accumulated *before* this step, so self-absorption within the step is not double
        // counted.
        const T g3 = g * g * g;
        emission[j] += density * line.kappa0 * line.profile(E_loc - line.rest_energy) * source
                       * exp(-absorption[j]) * dl_local / g3;
        absorption[j] += d_tau;
    }
}

// =================================================================================================
// FlatRayTransfer
// =================================================================================================

template <typename T>
FlatRayTransfer<T>::FlatRayTransfer(const RTField<T>& field, const LineTransition<T>& line, const SpectrumGrid<T>& bins)
    : m_field(field), m_line(line), m_bins(bins)
{
}

template <typename T>
void FlatRayTransfer<T>::trace_ray(T p, T z_obs, T z_far, T dz, T R_star, T I_star,
                                     std::vector<T>& emission, std::vector<T>& absorption) const
{
    const size_t n = m_bins.energy.size();
    emission.assign(n, T(0));
    absorption.assign(n, T(0));

    T g_metric[4][4];
    minkowski<T>(g_metric);
    T p_photon[4] = {1, 0, 0, 1};   // forward-travelling photon (towards +z, the observer), lab frame

    // The star (radius R_star) is an opaque photosphere, not a transparent cavity: for p < R_star the
    // (backward-traced) ray terminates at its near-side surface, at z_stop = +sqrt(R_star^2 - p^2) -- the
    // far-side wind, and the star's own interior/far hemisphere, are physically blocked and must not
    // contribute (otherwise this sightline would spuriously pick up far-side emission/absorption it can
    // never actually see, breaking photon conservation).
    const T z_stop = (p < R_star) ? sqrt(R_star*R_star - p*p) : z_far;

    for (T z = z_obs; z > z_stop; z -= dz)
    {
        const T r = sqrt(p*p + z*z);
        if (!m_field.in_wind(r, T(0), T(0))) continue;

        T et[4];
        m_field.flat_four_velocity(p, T(0), z, et);
        const T g = dot_product<T>(g_metric, et, p_photon);

        const T density = m_field.density(r, T(0), T(0));
        const T source = I_star * dilution_factor<T>(r, R_star);

        accumulate_step<T>(dz, g, density, source, m_line, m_bins, absorption, emission);
    }

    // transmitted stellar continuum, for lines of sight that intercept the photosphere directly
    if (p < R_star)
        for (size_t j = 0; j < n; j++)
            emission[j] += I_star * exp(-absorption[j]);
}

// =================================================================================================
// RayTransfer (Kerr / symplectic backend)
// =================================================================================================

// Re-take the composed Mino-time step of size `frac` from q_prev (a fresh stepper, since MinoStepper
// carries no state beyond a, k, h, order and the last step's hit_horizon flag).
template <typename T>
static void trial_step(const MinoState<T>& q_prev, T a, T k, T h, int order, T frac, MinoState<T>& q_out)
{
    q_out = q_prev;
    MinoStepper<T> st(a, k, h, order);
    st.step(q_out, frac);
}

// Bisect the full composed step [0, step] taken from q_prev to the fraction at which `crossed` first
// becomes true (60 iterations, ~1e-13 of the step) -- same pattern as CausticBundle::bisect_step and the
// boundary bisection inside Raytracer::propagate_symplectic_impl. Leaves q_out at that fraction.
template <typename T, class Pred>
static T bisect_boundary(const MinoState<T>& q_prev, T step, T a, T k, T h, int order, Pred crossed, MinoState<T>& q_out)
{
    T lo = 0, hi = step;
    for (int it = 0; it < 60 && (hi - lo) > T(1e-13) * step; it++)
    {
        const T mid = (lo + hi) / 2;
        MinoState<T> trial;
        trial_step<T>(q_prev, a, k, h, order, mid, trial);
        if (crossed(trial)) hi = mid; else lo = mid;
    }
    const T frac = (lo + hi) / 2;
    trial_step<T>(q_prev, a, k, h, order, frac, q_out);
    return frac;
}

template <typename T>
RayTransfer<T>::RayTransfer(const ImagePlane<T>& plane, T spin, const RTField<T>& field,
                              const LineTransition<T>& line, const SpectrumGrid<T>& bins,
                              int Nx, int Ny, T x0, T dx, T y0, T dy,
                              const RayDestination<T>* disc, const ContinuumSource<T>* corona,
                              WindSourceMode source_mode, T density_scale)
    : m_plane(plane), m_spin(spin), m_field(field), m_line(line), m_bins(bins),
      m_Nx(Nx), m_Ny(Ny), m_x0(x0), m_dx(dx), m_y0(y0), m_dy(dy),
      m_disc(disc), m_corona(corona),
      m_order(6), m_step(-1),
      m_max_tstep(MAXDT), m_max_phistep(MAXDPHI), m_maxtstep_rlim(MAXDT_RLIM),
      m_source_mode((corona != nullptr) ? source_mode : WindSourceMode::Density),
      m_density_scale(density_scale)
{
    const int n_energy = (int)m_bins.energy.size();
    continuum_map = std::make_unique<Array2D<T>>(m_Nx, m_Ny);
    tau_map = std::make_unique<Array2D<T>>(m_Nx, m_Ny);
    flux_cube = std::make_unique<Array3D<T>>(n_energy, m_Ny, m_Nx);
    spec_line.assign(n_energy, T(0));
    spec_total.assign(n_energy, T(0));
}

template <typename T>
bool RayTransfer<T>::trace_pixel(T x, T y, std::vector<T>& line_emission, std::vector<T>& absorption, T& continuum,
                                   T r_max, int steplim) const
{
    const size_t n = m_bins.energy.size();
    line_emission.assign(n, T(0));
    absorption.assign(n, T(0));
    continuum = 0;

    // Non-positive (the default) resolves against this plane's own distance -- see the header comment on
    // trace_pixel() for why a fixed literal default would be unsafe here.
    const T r_max_eff = (r_max > 0) ? r_max : T(1.1) * m_plane.get_dist();
    const int steplim_eff = (steplim > 0) ? steplim : SYMP_STEPLIM;

    Ray<T> ray;
    m_plane.init_ray(ray, x, y);
    if (!std::isfinite(ray.h) || !std::isfinite(ray.Q) || !std::isfinite(ray.theta) || !std::isfinite(ray.phi))
        return false;

    const T a = -1 * m_spin;   // ImagePlane traces backward by negating the spin (same convention as CausticBundle)
    const T ray_emit = m_plane.emit_energy(ray, T(0), true);

    MinoState<T> q;
    mino_init<T>(ray.r, ray.theta, ray.k, ray.h, ray.Q, ray.rdot_sign, ray.thetadot_sign, a, q.u, q.pu, q.ptheta);
    q.theta = ray.theta; q.phi = ray.phi; q.t = ray.t;
    MinoStepper<T> stepper(a, ray.k, ray.h, m_order);

    const T h0 = (m_step > 0) ? m_step : T(1) / PRECISION;
    const T r_cap = 2 * kerr_horizon<T>(m_spin);

    T r, sd;
    mino_r_from_u<T>(q.u, a, r, sd);

    for (int steps = 0; steps < steplim_eff; steps++)
    {
        if (r >= r_max_eff) return false;   // escaped, no continuum

        T tdot, phidot;
        const T hstep = mino_step_size<T>(r, q.theta, ray.k, ray.h, a, h0, r_cap,
                                           m_max_tstep, m_maxtstep_rlim, m_max_phistep, tdot, phidot);

        // wind accumulation at the pre-step position -- the entire wind contributes to the line
        // regardless of the ray's eventual fate (corona, disc, escape)
        if (m_field.in_wind(r, q.theta, q.phi))
        {
            T et[4];
            m_field.four_velocity(r, q.theta, q.phi, m_spin, et);

            // Q recomputed from the current canonical state (rather than the ray's initial Q, which is
            // conserved only to truncation error for the symplectic integrator -- same practice as
            // CausticBundle::disc_redshift, caustic_bundle.h)
            const T Qc = carter_Q<T>(q.theta, q.ptheta, ray.k, ray.h, a);
            const int rdot_sign = (q.pu >= 0) ? 1 : -1;
            const int thetadot_sign = (q.ptheta >= 0) ? 1 : -1;
            const T g = m_plane.ray_redshift(et, true, r, q.theta, q.phi, ray.k, ray.h, Qc,
                                              rdot_sign, thetadot_sign, ray_emit);

            const T rhosq = r*r + (m_spin*cos(q.theta))*(m_spin*cos(q.theta));
            const T d_lambda = rhosq * hstep;
            const T density = m_field.density(r, q.theta, q.phi);
            T source;
            switch (m_source_mode)
            {
                case WindSourceMode::Illumination:
                    source = m_corona->illumination(r, q.theta, q.phi, m_spin);
                    break;
                case WindSourceMode::PowerLaw:
                    source = m_powerlaw_norm * pow(r * sin(q.theta) / m_powerlaw_ref_r, -m_powerlaw_index);
                    break;
                default:
                    source = m_density_scale * density;
                    break;
            }

            accumulate_step<T>(d_lambda, g, density, source, m_line, m_bins, absorption, line_emission);
        }

        const MinoState<T> q_prev = q;
        const T theta_prev = q.theta;
        const bool was_in_corona = (m_corona != nullptr) && m_corona->contains(r, q.theta, q.phi);

        stepper.step(q, hstep);
        if (!std::isfinite(q.u) || !std::isfinite(q.theta) || !std::isfinite(q.pu) || !std::isfinite(q.ptheta))
            return false;
        if (stepper.hit_horizon)
            return false;   // shouldn't normally happen if the corona encloses the horizon (see ray_transfer.h)

        T r_new, sd_new;
        mino_r_from_u<T>(q.u, a, r_new, sd_new);

        // --- corona: transition from outside to inside, checked *before* the disc so that a disc-surface
        //     continuum source (e.g. DiscContinuumSource, continuum_source.h -- a zero-thickness annulus
        //     on theta = pi/2 detected via the crossing-aware contains() overload, using the fixed
        //     theta_prev captured here) sharing the very same equatorial crossing as a wider opaque disc
        //     (e.g. DiscWithISCODestination) takes priority within its own annulus, rather than the disc
        //     always blocking it first. For a genuine volume corona (SphericalContinuumSource) the
        //     crossing-aware overload just falls back to the pointwise test, so this reorder does not
        //     change which condition fires when the two regions are geometrically disjoint, as in
        //     ray_transfer_disc_wind.cpp -----------------------------------------------------------------
        const bool now_in_corona = (m_corona != nullptr) && m_corona->contains(r_new, q.theta, q.phi, theta_prev);
        if (now_in_corona && !was_in_corona)
        {
            MinoState<T> q_hit;
            bisect_boundary<T>(q_prev, hstep, a, ray.k, ray.h, m_order,
                                [&](const MinoState<T>& trial)
                                {
                                    T rt, sdt;
                                    mino_r_from_u<T>(trial.u, a, rt, sdt);
                                    return m_corona->contains(rt, trial.theta, trial.phi, theta_prev);
                                }, q_hit);

            T r_hit, sd_hit;
            mino_r_from_u<T>(q_hit.u, a, r_hit, sd_hit);
            T et[4];
            m_corona->four_velocity(r_hit, q_hit.theta, q_hit.phi, m_spin, et);
            const T Qc = carter_Q<T>(q_hit.theta, q_hit.ptheta, ray.k, ray.h, a);
            const int rdot_sign = (q_hit.pu >= 0) ? 1 : -1;
            const int thetadot_sign = (q_hit.ptheta >= 0) ? 1 : -1;
            const T g_corona = m_plane.ray_redshift(et, true, r_hit, q_hit.theta, q_hit.phi, ray.k, ray.h, Qc,
                                                      rdot_sign, thetadot_sign, ray_emit);
            // I_nu/nu^3 is a photon-transport invariant: I_obs/E_obs^3 = I_loc/E_loc^3, so for a source
            // flat (energy-independent) in its own rest frame, I_obs = I_loc * (E_obs/E_loc)^3 = I_loc/g^3
            // (g = E_loc/E_obs > 1 for redshifted light -- dimmer observed, as it must be)
            continuum = m_corona->intensity / (g_corona * g_corona * g_corona);
            return true;
        }

        // --- accretion disc: an opaque stopping surface, hit exactly by bisection like any other
        //     RayDestination boundary (see propagate_symplectic_impl) --------------------------------
        if (m_disc != nullptr && m_disc->reached(r_new, q.theta, q.phi, theta_prev))
            return false;   // blocked: no continuum; line_emission/absorption already accumulated stand

        r = r_new;
    }

    return false;   // step limit
}

template <typename T>
void RayTransfer<T>::run_raytrace(T r_max, int steplim, int show_progress)
{
    const int n_energy = (int)m_bins.energy.size();
    const int Nx = m_Nx, Ny = m_Ny;
    const T x0 = m_x0, dx = m_dx, y0 = m_y0, dy = m_dy;

    // Resolved once here (not once per pixel) -- see trace_pixel()'s header comment for the non-positive
    // ("use the default") convention.
    const T r_max_eff = (r_max > 0) ? r_max : T(1.1) * m_plane.get_dist();
    const int steplim_eff = (steplim > 0) ? steplim : SYMP_STEPLIM;

    std::fill(spec_line.begin(), spec_line.end(), T(0));
    std::fill(spec_total.begin(), spec_total.end(), T(0));
    continuum_total = 0;
    tau_corona_max = 0;
    n_corona = 0;

    Array2D<T>& cmap = *continuum_map;
    Array2D<T>& tmap = *tau_map;
    Array3D<T>& fcube = *flux_cube;

    long n_corona_local = 0;
    T continuum_total_local = 0;
    T tau_corona_max_local = 0;

    // Same ProgressBar/atomic-counter pattern as Raytracer::run_raytrace (raytracer.cpp): one shared counter
    // incremented per pixel traced (not per row), so the bar advances smoothly regardless of how the row
    // loop happens to be scheduled across threads. show_progress's sign selects the drawn bar vs. a plain
    // "done/total" line (ProgressBar's own convention); its magnitude is the update interval, in pixels.
    const long n_pixels = (long)Nx * Ny;
    ProgressBar prog(n_pixels, "Pixel", 0, (show_progress > 0));
    show_progress = abs(show_progress);
    long pixels_done = 0;

    // Pixels are independent (trace_pixel is const and touches no shared mutable state), so the row loop
    // parallelises directly -- each thread gets its own line_emission/absorption scratch vectors and a
    // private per-row partial spectrum, merged into the shared spec_line/spec_total under a critical
    // section once per row (cheap next to the tracing itself). continuum_map/flux_cube are written at
    // disjoint (ix, iy) indices per row, so no synchronisation is needed for those.
    #pragma omp parallel for schedule(dynamic) reduction(+:n_corona_local) reduction(+:continuum_total_local) \
        reduction(max:tau_corona_max_local)
    for (int ix = 0; ix < Nx; ix++)
    {
        std::vector<T> line_emission(n_energy), absorption(n_energy);
        std::vector<T> row_spec_line(n_energy, T(0)), row_spec_total(n_energy, T(0));

        const T x = x0 + (ix + T(0.5)) * dx;
        for (int iy = 0; iy < Ny; iy++)
        {
            const T y = y0 + (iy + T(0.5)) * dy;
            T continuum;
            const bool hit_corona = trace_pixel(x, y, line_emission, absorption, continuum, r_max_eff, steplim_eff);
            if (hit_corona) ++n_corona_local;

            cmap[ix][iy] = continuum;
            continuum_total_local += continuum;
            T tau_peak = 0;
            for (int j = 0; j < n_energy; j++)
            {
                const T flux = continuum * exp(-absorption[j]) + line_emission[j];
                fcube[j][iy][ix] = flux;
                row_spec_line[j] += line_emission[j];
                row_spec_total[j] += flux;
                if (absorption[j] > tau_peak) tau_peak = absorption[j];
            }
            tmap[ix][iy] = tau_peak;
            if (hit_corona && tau_peak > tau_corona_max_local) tau_corona_max_local = tau_peak;

            if (show_progress != 0)
            {
                long done;
                #pragma omp atomic capture
                done = ++pixels_done;
                if (done % show_progress == 0)
                {
                    #pragma omp critical
                    prog.show(done);
                }
            }
        }

        #pragma omp critical
        {
            for (int j = 0; j < n_energy; j++)
            {
                spec_line[j] += row_spec_line[j];
                spec_total[j] += row_spec_total[j];
            }
        }
    }
    prog.done();

    n_corona = n_corona_local;
    continuum_total = continuum_total_local;
    tau_corona_max = tau_corona_max_local;
}

// =================================================================================================
// explicit instantiation
// =================================================================================================

template struct LineTransition<double>;
template struct SpectrumGrid<double>;
template void accumulate_step<double>(double, double, double, double, const LineTransition<double>&,
                                       const SpectrumGrid<double>&, std::vector<double>&, std::vector<double>&);
template class FlatRayTransfer<double>;
template class RayTransfer<double>;
