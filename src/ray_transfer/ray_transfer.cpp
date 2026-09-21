/*
 * ray_transfer.cpp
 *
 *  See ray_transfer.h for the design.
 */

#include "ray_transfer.h"
#include <algorithm>

// =================================================================================================
// SphericalBetaWind
// =================================================================================================

template <typename T>
SphericalBetaWind<T>::SphericalBetaWind(T v_inf_, T v0_, T beta_exp_, T R0_, T R_out_, T n0_)
    : v_inf(v_inf_), v0(v0_), beta_exp(beta_exp_), R0(R0_), R_out(R_out_), n0(n0_)
{
}

template <typename T>
T SphericalBetaWind<T>::velocity(T r) const
{
    if (r <= R0) return v0;
    return v0 + (v_inf - v0) * pow(1 - R0/r, beta_exp);
}

template <typename T>
T SphericalBetaWind<T>::density(T r, T theta, T phi) const
{
    if (r <= R0 || r >= R_out) return 0;
    const T v = velocity(r);
    if (v <= 0) return 0;   // v0 == 0 (and hence v(r) == 0 everywhere): no wind material at all
    // mass continuity n(r) v(r) r^2 = const, normalised at r_ref = 2*R0
    const T r_ref = 2*R0;
    const T v_ref = velocity(r_ref);
    return n0 * (v_ref * r_ref*r_ref) / (v * r*r);
}

template <typename T>
void SphericalBetaWind<T>::four_velocity(T r, T theta, T phi, T spin, T et[4]) const
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

template <typename T>
void SphericalBetaWind<T>::flat_four_velocity(T x, T y, T z, T et[4]) const
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

// =================================================================================================
// Corona
// =================================================================================================

template <typename T>
void Corona<T>::four_velocity(T r, T theta, T phi, T spin, T et[4]) const
{
    // static (V = 0) observer -- same tetrad construction SphericalBetaWind::four_velocity uses at zero
    // boost, kept unboosted here (non-rotating corona, per docs/plan_ray_transfer.md)
    T pos[4] = {0, r, theta, phi};
    T e1[4], e2[4], e3[4];
    tetrad<T>(et, e1, e2, e3, pos, T(0), spin);
}

template <typename T>
T SphericalCorona<T>::illumination(T r, T theta, T phi, T spin) const
{
    // Geometric dilution: the same solid-angle formula FlatRayTransfer's star uses (dilution_factor,
    // above), treating the corona as an isotropically-emitting sphere.
    //
    // Redshift from the corona's radius out to this point: for a static (V = 0) observer, the locally
    // measured energy of a photon with conserved energy-at-infinity k is E = k/sqrt(g00) exactly, for
    // *any* Kerr spin and *any* connecting null geodesic -- g00*p^t + g03*p^phi = k identically for any
    // Kerr geodesic (k is this codebase's p_t, and p_t = g_{0 nu} p^nu trivially; verified directly
    // against momentum_from_consts + kerr_metric, not just derived -- docs/plan_ray_transfer.md Sec 5.11),
    // so this is independent of the photon's own h/Q and does not depend on which specific (generally
    // bent) path an illuminating photon takes between the two radii; it is the same static-to-static
    // Killing-energy argument the corona's own continuum boost and the Schwarzschild-redshift test
    // already rely on. Evaluating the corona and the point at the *same* theta is the one approximation
    // here (a radial illuminating path) -- the dilution factor above is the bigger one: unlike this
    // redshift ratio, it is *not* protected by the Killing-energy cancellation (solid angle depends on
    // how a bundle of geodesics spreads, which curvature/lensing does change), so it can miss lensing
    // magnification, additional bent-light images, and self-occultation -- see Sec 5.11 for why fixing
    // that would need a real ray trace from every wind point, not just two metric evaluations.
    T pos_corona[4] = {0, R_corona, theta, phi};
    T pos_point[4] = {0, r, theta, phi};
    T g_corona[4][4], g_point[4][4];
    kerr_metric<T>(g_corona, pos_corona, spin);
    kerr_metric<T>(g_point, pos_point, spin);
    const T g_illum = sqrt(g_point[0][0] / g_corona[0][0]);   // E_static(corona) / E_static(point)
    return this->intensity * dilution_factor<T>(r, R_corona) / (g_illum * g_illum * g_illum);
}

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
                              const RayDestination<T>* disc, const Corona<T>* corona,
                              int symp_order, T symp_step, T r_max, int steplim,
                              T max_tstep, T max_phistep, T maxtstep_rlim,
                              WindSourceMode source_mode, T density_scale)
    : m_plane(plane), m_spin(spin), m_field(field), m_line(line), m_bins(bins),
      m_disc(disc), m_corona(corona),
      m_order((symp_order == 2 || symp_order == 4) ? symp_order : 6),
      m_step(symp_step), m_r_max((r_max > 0) ? r_max : T(1.1) * plane.get_dist()), m_steplim(steplim),
      m_max_tstep(max_tstep), m_max_phistep(max_phistep), m_maxtstep_rlim(maxtstep_rlim),
      m_source_mode((corona != nullptr) ? source_mode : WindSourceMode::Density),
      m_density_scale(density_scale)
{
}

template <typename T>
bool RayTransfer<T>::trace_pixel(T x, T y, std::vector<T>& line_emission, std::vector<T>& absorption, T& continuum) const
{
    const size_t n = m_bins.energy.size();
    line_emission.assign(n, T(0));
    absorption.assign(n, T(0));
    continuum = 0;

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

    for (int steps = 0; steps < m_steplim; steps++)
    {
        if (r >= m_r_max) return false;   // escaped, no continuum

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
            const T source = (m_source_mode == WindSourceMode::Illumination)
                              ? m_corona->illumination(r, q.theta, q.phi, m_spin)
                              : m_density_scale * density;

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

        // --- accretion disc: an opaque stopping surface, hit exactly by bisection like any other
        //     RayDestination boundary (see propagate_symplectic_impl) --------------------------------
        if (m_disc != nullptr && m_disc->reached(r_new, q.theta, q.phi, theta_prev))
            return false;   // blocked: no continuum; line_emission/absorption already accumulated stand

        // --- corona: transition from outside to inside -----------------------------------------------
        const bool now_in_corona = (m_corona != nullptr) && m_corona->contains(r_new, q.theta, q.phi);
        if (now_in_corona && !was_in_corona)
        {
            MinoState<T> q_hit;
            bisect_boundary<T>(q_prev, hstep, a, ray.k, ray.h, m_order,
                                [&](const MinoState<T>& trial)
                                {
                                    T rt, sdt;
                                    mino_r_from_u<T>(trial.u, a, rt, sdt);
                                    return m_corona->contains(rt, trial.theta, trial.phi);
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

        r = r_new;
    }

    return false;   // step limit
}

// =================================================================================================
// explicit instantiation
// =================================================================================================

template struct LineTransition<double>;
template struct SpectrumGrid<double>;
template class RTField<double>;
template class SphericalBetaWind<double>;
template class ConicalBetaWind<double>;
template class Corona<double>;
template class SphericalCorona<double>;
template void accumulate_step<double>(double, double, double, double, const LineTransition<double>&,
                                       const SpectrumGrid<double>&, std::vector<double>&, std::vector<double>&);
template class FlatRayTransfer<double>;
template class RayTransfer<double>;
