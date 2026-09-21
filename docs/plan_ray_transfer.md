# `RayTransfer` — radiative transfer along traced rays

Plan (2026-09-20). Status: **implemented** (`src/ray_transfer/`); Kerr backend built but not yet applied to
a disc-wind science case; P-Cygni verification test passing.

## 1  Goal

Integrate the radiative transfer equation for a resonance line *along* a traced null geodesic, rather than
stopping at a single destination surface as `Raytracer`/`ImagePlane` do: accumulate an energy-dependent
optical depth (absorption) and an attenuated local emission contribution at every step, so that line
profiles from extended, moving material can be synthesized. Initial target: a P-Cygni profile from a
spherical wind (no black hole). Eventual target: X-ray line emission/absorption from a wind launched off an
accretion disc, viewed through the Kerr spacetime.

Constraints: no changes to `raytracer.h`/`raytracer.cpp` (only their public API is reused); new code lives
in its own library, `src/ray_transfer/`; the primary integrator is the existing symplectic (Mino-time)
stepper, but driven directly rather than through `propagate_symplectic`/`propagate_symplectic_impl` (those
are `protected` on `Raytracer<T>` and have no per-step hook) — the same situation `CausticBundle`
(`src/caustic/caustic_bundle.h`) was in, solved the same way: compose a `const ImagePlane<T>&`, use its
public `init_ray`/`emit_energy`/`ray_redshift`, and drive `MinoStepper`/`mino_step_size`
(`mino_stepper.h`) directly.

## 2  Physics

**Local (comoving) energy and g-factor.** The null-geodesic path a photon follows does not depend on its
energy — only the local, comoving frequency at each point (and hence the line opacity/emissivity it
experiences there) depends on energy, via the Doppler/gravitational shift between the photon and the local
material 4-velocity `u^mu`. That shift is `g = E_loc / E_obs`, with `E_loc = g_mn u^m k^n` (`k^mu` the
photon's null tangent) — exactly the dot product `Raytracer::emit_energy`/`ray_redshift` already compute,
just with the wind's 4-velocity substituted for a circular-orbit one. For a fixed energy grid `E_obs[j]`,
the local resonance energy at a step is `E_loc_j = E_obs[j] * g`.

**Proper path length.** Opacity/emissivity are rest-frame quantities of the material, so the length
entering the optical-depth integral is not the metric's `ds` (identically zero along a null geodesic,
`ds^2 = (k.k) dlambda^2 = 0`) but the distance swept through the local fluid element as measured by an
observer comoving with `u^mu`. Decomposing the null tangent as `k^mu = E_loc (u^mu + n^mu)` (`n^mu` the
local unit propagation direction) and projecting `dx^mu/dlambda = k^mu` onto the observer's proper-time and
proper-distance axes gives `dl_local/dlambda = E_loc`, i.e. `dl_local = g * dlambda` (the traced ray's own
`E_obs` is fixed, so this is common to every energy bin — computed once per step, not once per bin). This
is the standard "invariant path length" used in GR radiative-transfer codes.

**No black hole ≠ spin → 0.** `kerr.h` fixes the central mass to 1 geometric unit everywhere
(`kerr_horizon(0) = 2`, etc.) — there is no parameter that turns the Kerr metric into flat spacetime. The
P-Cygni verification case therefore needs a genuinely separate flat-space ray path (straight lines,
`minkowski()`/`dot_product()` from `kerr.h`), not `spin = 0` Kerr.

## 3  Class design (`src/ray_transfer/ray_transfer.h`, `.cpp`)

- `LineTransition<T>` — rest energy, Doppler width, opacity normalisation `kappa0`; normalised Gaussian
  `profile()`.
- `SpectrumGrid<T>` — fixed observer-frame energy bins.
- `RTField<T>` — abstract volumetric material field (velocity + density); deliberately not a
  `RayDestination<T>` subclass (that interface is a stopping *surface*, not a volume). Two velocity
  accessors, because the Kerr and flat backends need genuinely different tetrad constructions (not related
  by a `spin -> 0` limit): `four_velocity(r,theta,phi,spin,et)` (Boyer-Lindquist, for `RayTransfer`) and
  `flat_four_velocity(x,y,z,et)` (Minkowski Cartesian, for `FlatRayTransfer`).
- `SphericalBetaWind<T> : RTField<T>` — classic beta-velocity-law wind, `v(r) = v_inf (1-R0/r)^beta`,
  density from mass continuity (normalised at `2*R0`, since `v(R0)=0`). `four_velocity` boosts the static
  Kerr tetrad (`kerr_metric`, `tetrad`, both `kerr.h`) radially by `v(r)`; `flat_four_velocity` does the
  same in Minkowski Cartesian coordinates.
- `dilution_factor(r, R_star)` — geometric dilution of an isotropically-emitting sphere (Mihalas), used as
  the (single-scattering, optically-thin-illumination) local source function `S(r) = I_star * dilution_factor(r, R_star)`.
- `accumulate_step(d_lambda, g, density, source, line, bins, absorption&, emission&)` — the shared
  per-step core used identically by both backends: for every bin, `E_loc_j = bins[j]*g`,
  `d_tau = density*kappa0*profile(E_loc_j - E0)*g*d_lambda`; emission uses the optical depth accumulated
  *before* this step's own contribution (no self-absorption double-counting).
- `FlatRayTransfer<T>` — straight-line rays in Minkowski space, fixed photon `p^mu = (1,0,0,1)`, so
  `E_obs = 1` trivially and `g` is just the local dot product. `trace_ray(p, z_obs, z_far, dz, R_star,
  I_star, ...)` stops the (backward-traced) ray at the star's *near-side* photosphere for `p < R_star`
  (`z_stop = sqrt(R_star^2-p^2)`) rather than passing straight through — the star is opaque, so the far-side
  wind and the star's interior are not visible along that sightline (getting this wrong was caught by an
  early, more ambitious version of the verification test — §4).
- `RayTransfer<T>` — the Kerr/symplectic backward-traced version, composing `const ImagePlane<T>&` and the
  physical spin (passed explicitly, `Raytracer::spin` being `protected` — same as
  `CausticBundle::Params::from_plane`). `trace_pixel` builds the ray via `plane.init_ray`, converts to the
  canonical Mino state via `mino_init`, then runs its own loop calling `MinoStepper::step`/`mino_step_size`
  directly (mirroring `propagate_symplectic_impl`/`CausticBundle::trace`), computing `g` at each step via
  the *public* `Raytracer::ray_redshift(et[4], ...)` overload with `Q` recomputed from the current state via
  `carter_Q` (same practice as `CausticBundle::disc_redshift`, since `Q` is conserved only to truncation
  error for the symplectic integrator) and `dl_local`'s `d_lambda = (r^2+a^2cos^2theta) * hstep`.

## 4  Verification

`src/tests/ray_transfer_pcygni_test.cpp`, run against `FlatRayTransfer` with a beta-velocity-law wind:

- **Morphology**: blueshifted absorption trough (residual < 0.9 above the line energy) and redshifted
  emission excess (residual > 1.01 below it) — the defining P-Cygni signature.
- **Doppler-driven asymmetry**: with `v_inf = 0` the residual spectrum is exactly symmetric about the line
  energy (`g = 1` everywhere, no preferred side); with the wind flowing it becomes strongly asymmetric.

An exact photon-conservation identity (`integral(residual - 1) dE = 0`) was tried first and dropped: for a
single fixed observer direction and an extended, isotropically-scattering photosphere it is **not** an
exact invariant of this single-scattering model (the identity only closes once emission is integrated over
the full 4π sr of "virtual observers," not one sightline — worked out analytically by converting the
impact-parameter integrals to volume integrals: the absorption side samples one hemisphere's worth of
illumination geometry, the emission side samples the mean intensity `W(r)` at every point, and the two are
related by a factor of the *shadow* region's own contribution, which is not generally zero). It was also
numerically fragile — very sensitive to the ratio of the line's Doppler width to the line-of-sight step
size (confirmed by a standalone convergence check: the residual imbalance changed sign and by an order of
magnitude under both step-size and Doppler-width refinement, i.e. it is dominated by resonance-quadrature
aliasing, not signal). The asymmetry check above is both exact (down to floating point) and
implementation-risk-free, so it replaced it.

`src/ray_transfer/ray_transfer_pcygni.cpp` (par file `par/ray_transfer_pcygni.par`, template
`par_example/ray_transfer_pcygni.par_example`) sums the per-impact-parameter spectra into a
continuum-normalised residual profile, CSV to `dat/`; `python/ray_transfer_pcygni_plot.py` plots it.

## 5  Disc-wind application (Kerr backend)

Plan (2026-09-20), implemented same day: `src/ray_transfer/ray_transfer_disc_wind.cpp` — a spherical wind
bounded by inner/outer radii (`SphericalBetaWind::R_out`, added), a compact static `SphericalCorona` that
sources the continuum, and an equatorial `DiscWithISCODestination` (`ray_destination.h`, reused as is) that
blocks rays. `RayTransfer::trace_pixel` gained two optional pointers (`disc`, `corona`, both `nullptr` by
default) and now returns the continuum separately from the wind's line emission/absorption
(`continuum * exp(-absorption[j]) + line_emission[j]` is the observed flux per bin), since "only rays that
end on the corona contribute to the continuum" while "the entire wind contributes to the line emission"
regardless of the ray's eventual fate. Both new boundaries are hit exactly by bisecting the full composed
Mino-time step (`bisect_boundary` in `ray_transfer.cpp`) — the same convention
`propagate_symplectic_impl`/`CausticBundle` already use for `RayDestination` boundaries. The corona's
continuum is redshifted via `I_obs = I_corona * g_corona^3` (the standard `I_nu/nu^3` invariant for a
locally flat-spectrum source), `g_corona` computed through the same public `ray_redshift(et[4], ...)` reuse
the wind already used, just with `Corona::four_velocity`'s static observer instead.

**Bug found via the exact verification check below**: the constructor's original default `r_max = 1000`
collided with a realistic image-plane distance (also commonly 1000), so rays "escaped" on the very first
loop iteration without taking a single step. Fixed to default off `1.1 * plane.get_dist()`
(`CausticBundle::Params::from_plane`'s convention) when `r_max <= 0`.

**Verification** (`src/tests/ray_transfer_disc_wind_test.cpp`):
- *Schwarzschild static-corona redshift* — exact, independent formula: at `spin = 0`,
  `momentum_from_consts` collapses to `p^t = k/(1-2/r)` independent of the photon's angular momentum, so
  `E_loc(r) = k/sqrt(1-2/r)` for *any* ray that reaches the corona; checked to `2e-8` relative precision
  (limited only by the bisection tolerance) against the code's `continuum`/`g_corona` output.
- *Disc blocking* — a small image-plane scan at high inclination confirms both outcomes actually occur
  (some pixels blocked near the disc's midplane, some reach the corona) — a real end-to-end sanity check of
  the new termination logic, not just the redshift formula.
- A visual check (`python/ray_transfer_disc_wind_plot.py`) of a 40×40 test grid (spin 0.9, incl 60°, corona
  inside the ISCO) shows a continuum map shaped by the disc and photon-ring lensing/blueshift, and a wind
  line-flux slice away from the corona.

## 5.1  Image-plane-integrated spectrum, and two more bugs it caught

Asked to "check what the emission and absorption line spectrum looks like": `ray_transfer_disc_wind.cpp`
now also sums `continuum`/`line_emission`/`absorption` over every pixel and writes
`<outfile>_spectrum.csv` (`energy, line_flux, total_flux, residual`; `residual = total_flux /
continuum_total`, the disc-wind analogue of `ray_transfer_pcygni`'s residual flux), plotted by
`python/ray_transfer_disc_wind_spectrum_plot.py`. The first attempt (default parameters) produced a
spectrum with a residual spike of **10⁴–10⁷×** the continuum in a narrow energy range — not a plotting
artifact, two real bugs, found by not accepting an implausible number at face value:

1. **`SphericalBetaWind::density(r) ∝ 1/v(r)`, and `v(R0) = 0` exactly** (mass continuity), so density
   formally diverges (logarithmically, for `beta = 1`) right at the wind's launch radius. Fixed with a
   floor on `1 - R0/r` (`ray_transfer.cpp`, `velocity()`) so the divergence is bounded rather than literal
   — necessary, but on its own only reduced the spike by roughly a factor of the floor's inverse, not
   enough (confirmed the floor alone wasn't the dominant effect: tightening it 100× only shrank the peak
   ~8×).
2. **The dominant cause**: `RayTransfer`'s far-field Mino-time step was hard-coded to
   `raytracer.h`'s library-wide `MAXDT`/`MAXDT_RLIM`/`MAXDPHI` defaults, tuned for cheaply tracing rays
   through the near-vacuum far field of a plain disc image — far too coarse to resolve a wind whose density
   varies steeply close to its own launch radius `R0` (which, unlike the strong-field region near the
   horizon, sits *beyond* `r_cap` and so is entirely governed by this coarse far-field policy). A step
   landing close to `R0` samples the still-large-but-now-bounded density from bug (1) and aliases it into
   a spurious spike at whatever energy that step's local redshift happens to correspond to. Confirmed by a
   convergence scan (`max_tstep`: `1 -> 987`, `0.2 -> 58`, `0.05 -> 12.4`, `0.01 -> 8.6`, `0.005 -> 8.3`
   peak residual) — monotonic, converging, i.e. a genuine resolution artifact, not a physical divergence.
   Fixed by adding `max_tstep`/`max_phistep`/`maxtstep_rlim` as `RayTransfer` constructor parameters
   (previously hard-coded) exposed as a `max_tstep` par-file parameter, defaulted to `0.01` (converged to
   within ~10% of the finer-resolution value; tighter is more accurate but *much* more compute — a single
   40×40, 90-bin run went from ~1 s at the library default to ~95 s at `max_tstep = 0.01`).

With that fix, the spectrum is smooth and physically sensible for this (deliberately strong,
demonstration-only) parameter set: a broad, **redshifted emission** feature (up to ~11× continuum, around
4.4–5.3 keV against a 6.4 keV rest energy) from the slow-moving, dense inner wind plus gravitational
redshift, and a narrow, **near-saturated absorption trough** close to the rest energy (residual down to
~0.05 around 6.6 keV) from the near-corona wind material — a relativistically broadened/skewed analogue of
the flat-space P-Cygni shape, as expected.

## 5.3  Kerr vs flat: isolating gravitational redshift

`src/ray_transfer/ray_transfer_kerr_vs_flat.cpp` traces the *same* `SphericalBetaWind` + `SphericalCorona`
twice — once through `RayTransfer` (spin = 0 by default, i.e. Schwarzschild, so frame dragging doesn't mix
in) with no disc (keeps the system spherically symmetric, so the image plane's inclination is irrelevant),
once through `FlatRayTransfer` with the corona playing the role of `FlatRayTransfer::trace_ray`'s "star" —
and writes both continuum-normalised spectra to one CSV for direct overlay
(`python/ray_transfer_kerr_vs_flat_plot.py`). This is the wind-accumulation-path cross-check flagged as
missing in an earlier pass of this doc, prompted directly by being asked to isolate gravitational
redshift: with `R0 = 5`, `R_corona = 3` (horizon at `r = 2`), the flat spectrum's line sits close to the
6.4 keV rest energy (shifted only by the wind's own `v_inf = 0.05`), while the Kerr spectrum's whole
feature is dragged down to ~5 keV — the difference being gravitational redshift, nothing else, since every
other input (wind, corona, line) is identical between the two runs.

**A second normalisation bug found in the process**: the Kerr side self-normalises against its own
discretised pixel sum (`kerr_continuum_total`, summed over the same finite image-plane grid used
everywhere else), but the flat side was first normalised against the *exact* analytic continuum
`pi*R_corona^2*I_corona` — and the corona/star's radius (3) is small against the impact-parameter grid
sized for the wind's much larger extent (`R_out = 100`), so that exact value didn't match what the same
coarse `dp` grid actually integrates, leaving a spurious ~13% baseline offset far from the line on the
flat side only. Fixed by computing `flat_continuum_total` from the identical `p`/`dp` grid instead (summing
`weight*I_corona` for `p < R_corona` alongside the main loop), so the same discretisation error cancels in
both the numerator and denominator, exactly how the Kerr side already worked. Both baselines now sit at
exactly 1.0 far from the line, as they must.

Only 4 of 3600 Kerr pixels actually terminate on the corona at the example resolution (`Nx = 60`) — the
photon-capture cross-section (`b_crit = 3*sqrt(3) ≈ 5.2` at `a = 0`) is small against the pixel grid
spacing needed to also cover the wind's full 100 Rg extent. This mainly affects the *normalisation*
(quantifying how much of the small capture disc's area the coarse grid actually samples), not the
*redshift* itself: at `a = 0`, `g` depends only on `r`, so every captured ray crosses the corona at
essentially the same `g_corona` regardless of impact parameter — the energy at which the Kerr feature
appears is robust even with few samples; a production comparison would want a finer grid (or one focused
near the capture radius) for a less quantisation-noisy amplitude.

## 5.4  A third bug, found by not trusting a too-sharp peak

Asked to double check the Kerr solution because "a lot of the rays are piling up into a very sharp peak
just below 5 keV": tracing the exact ray responsible (impact parameter `b = 2.83`, one of the 4 pixels
above) step by step showed why. `SphericalBetaWind::density(r) ~ 1/v(r)` (mass continuity), and the
Sec 5.1 fix only floored `v(r)` numerically once `1 - R0/r` fell below a small cutoff — but that floor
creates a **thin shell just above `R0` where density plateaus at a large, and still dominant, value**, and
the ray spends *several* consecutive (correctly small) steps crossing that whole shell, each picking up a
large contribution at nearly the same redshift (`r` barely changes across the shell). That is a real
integral over a real plateau, not quadrature noise, which is why (unlike Sec 5.1) tightening the step size
did not shrink it — and tightening the floor *further* made it worse: working out the integral
analytically, a `beta = 1` velocity law makes `integral n(r) dr` near `R0` a **marginally,
logarithmically-divergent** quantity, so the contribution left over after any hard floor at width `eps` is
`~ -ln(eps)` — it *grows*, slowly, as the floor tightens, confirmed empirically (peak residual 987 -> 58 ->
21 -> 12.4 -> 8.6 -> 8.3 as `max_tstep` — which is what actually controls how finely the floored shell
gets resolved here — was tightened from 1 to 0.005; converging, but to a value set by the floor, not by
the true model).

The floor was treating the symptom (bounding `1/v(r)` after it starts blowing up) rather than the cause (a
wind that decelerates to *exactly* zero at its launch radius). Fixed by giving `SphericalBetaWind` a real,
required base velocity `v0 > 0` at `r = R0` (`v(r) = v0 + (v_inf - v0)(1-R0/r)^beta`, replacing the old
floored formula) — a real wind has some nonzero thermal/turbulent launch speed anyway, so this isn't a
numerical fudge, it's filling in a parameter the model was missing. `v0` is now a required constructor
argument (`v0_vinf` in the par files, as a fraction of `v_inf`); every existing call site
(`ray_transfer_pcygni[_test]`, `ray_transfer_disc_wind[_test]`, `ray_transfer_kerr_vs_flat`) was updated,
defaulting to `0.01 * v_inf` except where a test deliberately wants the fully static (`v0 = 0` too, no
wind material at all) degenerate case.

**Confirmed genuinely fixed, not just moved again**: re-running the exact same ray with `max_tstep` from
0.01 down to 0.001 now gives 332 -> 325 -> 325 -> 324 — flat, i.e. actually converged, not sliding towards
a floor-dependent number. What's left is real: for `beta = 1`, the leftover near-`R0` contribution scales
as `~ln(v_inf/v0)`, so it only shrinks *logarithmically* as `v0` is raised (doubling `v0_vinf` roughly
halves it, empirically) — a strong, narrow feature from a wind whose base is this dense and this slow is a
genuine property of these parameters, not a bug, but `ray_transfer_kerr_vs_flat.par_example`'s `kappa0 =
2.0` (copied from `ray_transfer_disc_wind.par_example`, tuned for a different purpose — showing a strong
line in its own right) saturated it to the point of dominating the whole comparison. Retuned the example
to `kappa0 = 0.2`, `v0_vinf = 0.05`, and `Nx = 120` (finer image-plane grid, spreading the capture-disc
rays' slightly different redshifts across several bins instead of stacking every one of only 4 identical-
by-symmetry rays into a single bin) — the feature is now smooth and physically credible (peak ~8x
continuum, decaying over many bins) rather than an isolated spike, with the same gravitational-redshift
conclusion. `ray_transfer_disc_wind.par_example` still uses the original, more saturated parameters
(`kappa0 = 2.0`, `v0_vinf` defaulting to `0.01`) — numerically fine now (converged, not floor-dependent),
but a production run of that application should consider the same retuning if a less saturated line is
wanted there too.

## 5.5  `R0` and disc-wind's own spectrum are not exempt

Two follow-up questions worth recording, both with counter-intuitive (checked, not guessed) answers.

**Does raising `R0` (moving the wind's launch radius further from the black hole) help?** No — it makes
it worse, and by a lot. Holding every other wind parameter fixed (including `R_out`, to rule out "more
total wind material" as the explanation) and re-tracing the exact same captured ray: `R0 = 3.5 -> 5 -> 10
-> 20 -> 50 -> 100` gives peak optical depth `~12 -> 15 -> 39 -> 94 -> 273 -> 319` — growing roughly in
proportion to `R0`. The shell of enhanced density sits at a fixed *fraction* of `R0` above the launch
point, so its absolute width in `r` (and the symplectic integrator's far-field step policy, which is
designed to keep the radial step a fixed fraction of `r` at large radii) both scale up with `R0`, not down.
The direction that actually helps is smaller `R0` (closer to the corona) — consistent with `R0 = 5`,
`R_corona = 3` already being on the "better" end of this trend, not the worse one.

**Why didn't `ray_transfer_disc_wind` show the same obvious spike, if it uses the same wind model?**
Because its par file was never actually tuned to avoid the effect — it still uses the original `kappa0 =
2.0`, `v0_vinf = 0.01` (the *worst* combination identified in Sec 5.4), not `ray_transfer_kerr_vs_flat`'s
retuned `0.2`/`0.05`. Directly checked (`spin = 0.9`, its own par's exact parameters): every one of the 46
corona-hitting pixels at the example resolution has the *same* huge per-ray values (`max_tau` 280–600,
`max_em` 37–132) as the one pixel that produced the obvious spike in the `spin = 0` comparison. The reason
it looked smooth rather than spiky is `spin = 0.9` (frame dragging) plus the range of image-plane positions:
unlike the `spin = 0` case, no two of those 46 rays are related by an exact symmetry, so each gets a
slightly different redshift and the same large contributions land in *different*, adjacent energy bins
instead of stacking into one — smearing what is mechanically the same artifact into something that reads
as a smooth, physically plausible broad line. It never was exempt; it was just disguised by having more
distinct trajectories to spread the effect across.

Re-tuned `ray_transfer_disc_wind.par_example` to `v0_vinf = 0.05`, `kappa0 = 0.2` (matching
`ray_transfer_kerr_vs_flat`) and re-ran it: the peak dropped from ~11.3x continuum to ~4.2x, and the trough
from ~0.05x to ~0.39x — a real, substantial reduction, not a rounding difference, confirming the earlier
disc-wind spectrum shown was inflated by the same mechanism.

**On "the two applications should give the same results"**: they are not set up to match on purpose, even
now — `ray_transfer_kerr_vs_flat` uses `spin = 0` and no disc specifically so the corona+wind system stays
spherically symmetric (isolating gravitational redshift from frame dragging and disc geometry), while
`ray_transfer_disc_wind` uses `spin = 0.9` with a disc as its actual science case. Both now use the same
retuned wind parameters (`v0_vinf = 0.05`, `kappa0 = 0.2`), so the *wind* physics is on equal, trustworthy
footing between them, but their spectra are still expected to differ (different spin, no disc vs. a disc,
different `Nx`/grid) — matching wind parameters is not the same claim as matching results.

## 6  Open items / later
- (Resolved, Sec 5.10) The default wind source function (`S = density`, in `RayTransfer::trace_pixel`) was
  a placeholder with no tie to the corona's actual brightness; `WindSourceMode::Illumination`
  (`Corona::illumination`) now provides a properly illumination-based `S(r)`, selectable per run.
- `accumulate_step`'s emission always assumes the illuminating field is optically thin (no self-shielding of
  the illuminating beam itself) — worth revisiting for a strong-lined disc wind where the ionizing
  continuum's own attenuation through the wind may matter.
- The `max_tstep = 0.01` default (Sec 5.1) is a pragmatic, not fully converged, choice for a reasonably
  fast demo; a production run should re-check convergence (tighten `max_tstep` until the spectrum stops
  changing) for its own specific wind/corona parameters rather than trusting this default blindly. A
  properly adaptive step policy near `R0` (rather than a single global cap) would be a more principled fix.
- (Sec 5.15) Line emission is computed along one geodesic per pixel, self-shielded by the *cumulative*
  optical depth on that same ray (`accumulate_step`'s `exp(-absorption[j])`, never reset within
  `trace_pixel`) — correct for that single ray, but it means any near-side column thick enough to produce
  a visible absorption trough also erases essentially all far-side (redshifted) emission carried on the
  same ray, so the classic symmetric P-Cygni emission bump can only appear in the optically-thin limit.
  Getting a real redshifted emission bump back would need line emission integrated over solid angle/volume
  (Sobolev- or Monte-Carlo-style), not one ray per pixel — a substantially bigger change, not attempted here.
- (Sec 5.19) Sec 5.18's `R0 = 200` run was tuned to global peak `tau = 1` (no corona hits were possible at
  that resolution to measure `tau_corona_max` directly), but Sec 5.19 showed the two can differ by `~9x`
  at `R0 = 50` — so that run's *true* `tau_corona_max` is unknown and could be well below `1`. Re-run
  `R0 = 200` at a resolution/field-of-view combination that actually gets corona hits (accepting the
  larger grid cost, Sec 5.18's `~12/fov^2` scaling) if an accurate `tau_corona_max = 1` comparison at that
  `R0` specifically is wanted.

## 5.2  Parallelising the pixel loop

`ray_transfer_disc_wind.cpp`'s row loop (`ix`) is now `#pragma omp parallel for schedule(dynamic)`
(matching `Raytracer::run_raytrace`'s own `schedule(dynamic)` choice, raytracer.cpp) — pixels are
independent (`RayTransfer::trace_pixel` is `const` and touches no shared mutable state: `ImagePlane`,
`RTField`, `Corona`, `RayDestination` are all read through `const` methods with no internal mutable
state). Each thread gets its own `line_emission`/`absorption` scratch vectors and a private per-row partial
spectrum; `continuum_map`/`flux_cube` are written at disjoint `(ix, iy)` indices so need no
synchronisation, while the scalar `n_corona`/`continuum_total` reductions and the shared `spec_line`/
`spec_total` arrays (merged once per row under `#pragma omp critical`, negligible next to the tracing
itself) are the only cross-thread coordination needed. `ray_transfer`'s `CMakeLists.txt` already links
`raytracer`, which links `OpenMP::OpenMP_CXX` `PUBLIC`, so no build changes were needed.

Verified both for speed and correctness on the 40×40/90-bin test case: 8 threads gave a 7.3× wall-clock
speedup (97 s → 13 s), and the CONTINUUM/FLUX FITS extensions and the spectrum CSV came out bit-identical
between an `OMP_NUM_THREADS=1` and an 8-thread run.

## 5.6  Retuning the original P-Cygni test/application too

The P-Cygni case (`ray_transfer_pcygni[_test]`, Sec 4) has the same near-launch density pileup (Sec 5.4-5.5)
as every other application of `SphericalBetaWind` — and arguably worse exposure to it, structurally: the
star's radius equals the wind's launch radius (`R_star = R0`, by design — the wind starts at the stellar
surface), so *every* direct-continuum ray (any `p <~ R0`) necessarily terminates right at the edge of the
enhanced-density shell, not just some special grazing subset. Checked directly (isolated rays, at the
original `kappa0 = 5.0`, `v0_vinf = 0.01`): `max_tau` up to `~145,000` for the ray at `p = R0` exactly, and
`~11,000`-plus even at `p = 0.5`, nowhere near grazing.

Despite that, the *summed* spectrum's trough only reached `~0.016`, not something astronomically smaller —
because different `p` put their resonance at different observed energies (the same smearing mechanism as
Sec 5.5, but from the continuous spread of impact parameters rather than from `spin` breaking a discrete
symmetry). More telling: scanning `kappa0` from `5.0` down to `~0.1` left the trough depth almost
unchanged (`0.016 -> 0.024`) — the profile was saturated by the pileup, not responding to `kappa0` as
physically intended, until `kappa0` dropped below that saturation floor. Retuned to `kappa0 = 0.002`,
`v0_vinf = 0.05` (both the par file and the test's own hard-coded values): trough `0.016 -> 0.54`, emission
bump `4.17 -> 1.33` — a real, unsaturated, kappa0-sensitive P-Cygni profile, with all of Sec 4's checks
still passing with comfortable margin (`asymmetry = 9.8`, well above the `1.0` threshold).

`ray_transfer_disc_wind_test.cpp`'s two wind constructions both use `n0 = 0` (wind switched off, to
isolate the redshift formula and disc-blocking logic from the wind's density model), so their `v0`/`kappa0`
values are currently inert either way — updated them to the same `v0_vinf = 0.05`, `kappa0 = 0.2` used
everywhere the wind is actually on anyway, for consistency and to avoid confusion if a future edit turns
`n0` on without noticing the stale values.

(Superseded: the "no OpenMP" limitation noted for `RayTransfer`'s per-pixel loop in an earlier pass of this
doc's open-items list was fixed in Sec 5.2.)

## 5.7  Matching `ray_transfer_kerr_vs_flat`'s Kerr side to `ray_transfer_disc_wind`

The wind parameters (`v_inf`, `v0_vinf`, `beta`, `R0`, `R_out`, `n0`, `R_corona`, `I_corona`, `kappa0`,
`doppler_width`, `line_energy`) already matched between the two par files (Sec 5.6). `spin` and the
image-plane geometry did not: `ray_transfer_kerr_vs_flat.par_example` used `spin = 0` (deliberately, to
isolate gravitational redshift from frame dragging — Sec 5.3) and a grid sized to the wind's full extent
(`x0 = -1.2*R_out`, `Nx = 120`), while `ray_transfer_disc_wind.par_example` uses `spin = 0.9`,
`incl = 60` and a much smaller, finer grid (`x0 = -20`, `Nx = 60`). Matched `spin`, `dist`, `incl`, `x0`,
`xmax`, `Nx` and `max_tstep` to `ray_transfer_disc_wind.par_example`'s values (the `spin = 0` comparison
in Sec 5.3 stays as its own thing — this is a second, differently-purposed run of the same application).

Re-ran: the Kerr spectrum's peak and trough now land at essentially the same energies as
`ray_transfer_disc_wind`'s own spectrum (peak `~4.9` keV vs. `~4.9` keV; trough `~6.5` keV vs. `~6.6` keV),
and the same order of magnitude (peak `2.5x` vs. `4.2x` continuum; trough `0.32x` vs. `0.39x`) — but not
identical. The remaining gap is structural, not a bug: `ray_transfer_kerr_vs_flat.cpp` has no disc
parameter at all (it always passes `disc = nullptr` to `RayTransfer`), so pixels whose rays would have
been blocked by `ray_transfer_disc_wind`'s `DiscWithISCODestination` (inner edge at the ISCO, `r_out_disc
= 50`) instead continue past where the disc would have stopped them, picking up additional wind material
(and, for some, still reaching the corona) that `ray_transfer_disc_wind` never counts for those same
pixels. Getting bit-for-bit identical output between the two would need `ray_transfer_kerr_vs_flat.cpp` to
grow the same disc option `ray_transfer_disc_wind.cpp` has — not attempted here, since it isn't needed for
either application's own stated purpose (an isolated-redshift comparison, and a disc-wind science
calculation), only for a byte-for-byte cross-check between them.

Setting `spin = 0.9` also changes what the comparison *demonstrates*: with frame dragging back in, the gap
between the Kerr and flat spectra is now gravitational redshift *and* frame dragging together, not
gravitational redshift in isolation — the `spin = 0` version (Sec 5.3) is still the right configuration
for that specific claim.

## 5.8  Closing the gap: disc support added, exact match confirmed

Followed up by giving `ray_transfer_kerr_vs_flat.cpp` the same disc option `ray_transfer_disc_wind.cpp`
has: a `DiscWithISCODestination` (`r_isco = kerr_isco(spin, +1)`, outer edge `r_out_disc`), passed to
`RayTransfer` in place of the `nullptr` it always used before. `r_out_disc <= 0` (the default) keeps the
old, disc-free behaviour, so the `spin = 0` isolated-redshift comparison (Sec 5.3) is unaffected; set it
`> 0` (now `50.0` in `ray_transfer_kerr_vs_flat.par_example`, matching `ray_transfer_disc_wind.par_example`)
to enable it. The flat side has no disc equivalent (`FlatRayTransfer` has no such concept), so this only
changes the Kerr side, as intended.

Re-ran with the disc enabled: `ray_transfer_kerr_vs_flat`'s Kerr spectrum now matches
`ray_transfer_disc_wind`'s own spectrum essentially exactly — peak `4.220x` at `4.876` keV vs. `4.223x` at
`4.871` keV; trough `0.396x` at `6.545` keV vs. `0.391x` at `6.590` keV. The residual, sub-percent
difference is just the two par files' energy grids having slightly different bin centers (`energy_min`/
`energy_max` differ between them), not a physics discrepancy — closing the gap identified in Sec 5.7 was
purely a matter of the missing disc option, confirmed by adding it.

## 5.9  Fourth bug: the corona's `g^3` had the wrong sign, and the wind's emission was missing it entirely

Asked directly whether the wind's line emission carries the `g^3`-type invariant-intensity weighting the
corona's continuum does. It didn't — `accumulate_step`'s emission term had no such factor at all, only the
single (unrelated) power of `g` already inside `dl_local = g*dlambda`. Worse, checking the corona's own
`g^3` factor against the invariant it's supposed to implement (`I_nu/nu^3` conserved along a ray in vacuum
— the standard formulation in GRRT codes, e.g. Dexter 2016) showed it had the **wrong sign**:

```
I_obs/E_obs^3 = I_loc/E_loc^3   =>   I_obs = I_loc * (E_obs/E_loc)^3 = I_loc * g^-3   (g = E_loc/E_obs)
```

`g > 1` means redshifted light (climbing out of the well) — that must come out *dimmer*, not brighter.
Sanity check against relativistic beaming (the flat-space limit of the same invariant): an approaching
source (`g < 1`) should look brighter, `g^-3 > 1` ✓; a receding one (`g > 1`) should look dimmer, `g^-3 < 1`
✓. The code had `continuum = I_corona * g_corona^3` — backwards, making near-critical, highly-redshifted
photon-ring rays anomalously *brighter* the more redshifted they were (this is very likely the actual
source of the "continuum brightening near the photon ring" noted as merely "physically plausible" back in
the first disc-wind run — it wasn't).

**Why the existing Schwarzschild-redshift test didn't catch this**: it checked that `g` extracted from
`continuum` (via `(continuum/I_corona)^(1/3)`) matched the closed-form `1/sqrt(1-2/r)` — which only
verifies that `ray_redshift` returns the *correct ratio* `g`, not that raising it to the *correct power*
(`+3` vs. `-3`) is being used to turn that ratio into an intensity boost. A self-consistent-but-wrong
formula passes a test built the same way. Fixed both:
- `RayTransfer::trace_pixel`: `continuum = I_corona / g_corona^3` (`ray_transfer.cpp`).
- `accumulate_step`: emission now divided by `g^3` as well (derived from the invariant transfer equation
  `d(I_nu/nu^3)/dlambda = j_nu/nu^2 - nu*alpha_nu*(I_nu/nu^3)`, converted to `dl_local` and back to
  `I_obs`; full derivation in the code comment). Updated the test's extraction formula to
  `(I_corona/continuum)^(1/3)` to match — it still validates to `2e-8` relative precision, confirming the
  fix didn't touch the redshift calculation itself, only how it's used.

**Effect on the demos**: `ray_transfer_pcygni` barely moved (peak `1.326 -> 1.325`, trough `0.541 -> 0.544`)
— `v_inf = 0.01` keeps `g` close to `1` throughout that wind, so `g^3` vs `g^-3` is a small correction
there. `ray_transfer_disc_wind` and `ray_transfer_kerr_vs_flat` changed substantially — `g` reaches `1.3`
to several near the black hole, where `g^3` vs `g^-3` differs by an order of magnitude or more: continuum
map max dropped from `~9.5` to `~1.8` (no more anomalous brightening), and the spectrum's peak grew
(`4.2x -> 23.4x`, since the continuum baseline it's normalised against shrank far more than the line
emission did) while the trough nearly disappeared (`0.39x -> 0.96x`). `ray_transfer_kerr_vs_flat`'s Kerr
side still matches `ray_transfer_disc_wind` almost exactly after the fix (`23.43x` vs. `23.37x` peak) —
the two applications remain mutually consistent, as they must, since both call the same corrected code.

## 5.10  The missing absorption feature: a wind source-function gap, not a bug

Asked why the (now correctly `g^-3`-boosted) Kerr disc-wind spectrum no longer showed the total flux
dropping *below* continuum on the blueshifted side — only a smaller emission excess there, no real
absorption trough. Checked the two additive pieces (`continuum*exp(-tau)` and `line_emission`) separately
at a blueshifted bin rather than guessing: the absorbed-continuum piece dropped from `38.6` (far from the
line) down to `7.5` near resonance — a genuine `>80%` attenuation, so `exp(-tau)` was working exactly as
it should. But `line_emission` at that same bin was `227.9` — about `30x` the `~31` units of continuum
that had actually been removed, completely swamping the real trough.

Root cause: `RayTransfer::trace_pixel`'s line source function was `source = density` — a placeholder
already flagged in Sec 6's open items, with no tie at all to how brightly the corona actually illuminates
that point. It happened to look roughly balanced before Sec 5.9's fix only because the corona was then
anomalously *bright* (the `g^3` sign bug); once the corona was correctly dimmed, the uncalibrated
placeholder emission — never tied to the corona's brightness in the first place, so it never dimmed with
it — dominated everywhere and hid the trough. Not a bug in the absorption/attenuation mechanism itself
(confirmed working correctly above); a genuine physics gap that the other fix made visible.

Implemented both a proper fix and a quick option, selectable per run:
- **`WindSourceMode::Illumination`** (new default): `Corona::illumination(r,theta,phi,spin)`
  (`SphericalCorona`, `ray_transfer.cpp`) — the corona's specific intensity, geometrically diluted
  (`dilution_factor`, the same solid-angle formula `FlatRayTransfer`'s star uses) and redshifted out to
  that point. The redshift factor is *exact*, not approximate, for any Kerr spin (derivation, numerical
  verification against the code, and an honest accounting of what *isn't* exact here — the dilution
  factor — in Sec 5.11).
- **`WindSourceMode::Density`** (opt-in): the original `density_scale * density`, unchanged, for a quick
  look without the illumination model.

Both selectable via `source_mode` (`illumination`/`density`) and `density_scale` in
`ray_transfer_disc_wind.par`/`ray_transfer_kerr_vs_flat.par` — `RayTransfer`'s C++ default stays `Density`
for backward compatibility with existing call sites that don't specify it (the `disc_wind_test` blocks,
whose wind is switched off anyway); the two applications' par files opt into `Illumination` explicitly.

Re-ran both with the fix: `ray_transfer_disc_wind`'s spectrum now shows a proper absorption trough
(`0.10x`-`0.15x` continuum across `~5.2`-`6.6` keV) with only a small emission excess (`1.007x` at `4.47`
keV) — the classic P-Cygni-style shape that had been missing. `ray_transfer_kerr_vs_flat`'s Kerr side
still matches it essentially exactly (`1.0075x`/`0.1028x` vs. `1.0071x`/`0.1025x`), even run at a coarser
image-plane grid (`60x60` vs. `100x100`) — the illumination-based source is smooth, so it converges well
without needing the fine sampling the corona-hit continuum piece alone required.

## 5.11  `Corona::illumination`'s redshift: what's exact, what's a shortcut, and why

Asked, reasonably, how `illumination()` can know the corona's brightness at a wind point *without* a
separate ray trace from the corona to every such point — since that's exactly what the observer's own
rays need a full trace for. The answer splits the calculation into two pieces with very different
footing: the redshift factor is exact and needs no ray trace at all; the geometric dilution factor is a
flat-space shortcut that does.

**Why the redshift needs no ray trace.** `p_t = g_{0mu} p^mu = g00*p^t + g03*p^phi` is a coordinate
identity (Boyer-Lindquist coordinates have no `g01`/`g02` terms), and this codebase's `k` *is* `p_t` by
construction — `momentum_from_consts` (`kerr.h`) is built so that identity holds for any `h`, `Q`, or
direction signs. Checked this directly against the actual code (not merely asserted): computing
`g00*p^t + g03*p^phi` via `momentum_from_consts` + `kerr_metric` at several `(r, theta, a, h)` combinations,
including `a = 0` and `a != 0`, gives exactly `k` in every case (`1.0` for `k = 1`, to full double
precision). For a static (`V = 0`) observer, `et = (1/sqrt(g00), 0, 0, 0)` (the normalised timelike
Killing vector), so the energy that observer measures is

```
E_static = g_mn et^m p^n = et^0 * (g00*p^t + g03*p^phi) = k / sqrt(g00)
```

— using the verified identity directly, with no dependence on which particular geodesic the photon
followed (its own `h`/`Q` never enter). Since `k` is conserved along the *whole* geodesic by definition,
this holds at every point on it. For two static observers on the *same* photon's path (corona and wind
point), the ratio is then

```
g_illum = E_static(corona) / E_static(point) = sqrt(g00(point) / g00(corona))
```

with `k` cancelling — the same ratio for *any* photon connecting the two points, regardless of its `h`/`Q`
or how many times it winds around the black hole. That is the whole reason this piece needs only two
`kerr_metric` evaluations (`SphericalCorona::illumination`, `ray_transfer.cpp`) rather than a ray trace: a
ray trace would be needed to find out whether, and via how many paths, two points are connected at all,
but *given* that a connecting geodesic exists, its redshift is fixed by the endpoints alone.

This is the exact same construction already validated (not just derived) in Sec 5.4's Schwarzschild test
— `g_expected = sqrt(1-2/dist)/sqrt(1-2/R_corona)` is `sqrt(g00(dist)/g00(R_corona))` at `a = 0`, checked
to `2e-8` there against the closed-form value. `illumination()`'s `g_illum` is the identical formula with
the wind point's `r` standing in for the observer's `dist`.

**Why the dilution factor is *not* similarly protected, and does need a ray trace to get right.**
`dilution_factor(r, R_corona)` is the flat-space solid-angle formula: how much of the sky a sphere of
radius `R_corona` covers as seen from `r`, assuming straight-line sight lines. Nothing about the Killing
energy argument above rescues this piece, because solid angle is a statement about how a *bundle* of
nearby geodesics spreads or focuses (geodesic deviation), not about the energy of one geodesic — and
that spreading is exactly what curvature changes. In particular the flat formula:
- ignores gravitational lensing changing the corona's apparent size (typically magnifying it, especially
  near the photon sphere);
- counts only the single "direct" near-side image, missing any additional images light bending around the
  black hole could bring into view;
- cannot detect partial self-occultation by the black hole itself for some point/corona geometries.

Getting this right *would* need the kind of calculation the question was pointing at: for every wind
sample point along every traced ray, trace a bundle of geodesics back toward the corona to find how much
of it (and how many lensed copies) is actually visible from there — the same class of problem the main
calculation already solves once, for the observer's own rays, but now needed at every step of every ray
instead of once per pixel. Left as a known, explicit approximation (decided not to pursue further for
now) rather than attempted here.

## 5.12  Absorption-dominated, weak-emission spectrum: geometry, not a bug

Asked why, with the illumination-based source now in, `ray_transfer_disc_wind`'s spectrum shows a deep
absorption trough but almost no visible emission bump (only `1.007x` continuum at the peak) — whereas the
flat-space P-Cygni test shows both features comparably. Checked how the *emission weight*
(`density(r) * illumination(r)`, the quantity that actually sets the local emission contribution) falls
off with `r`, against density alone (all that matters for absorption), for the example parameters
(`R0 = 5`, `R_corona = 3`, `R_out = 100`):

| `r` | density | illumination | density * illumination |
|---|---|---|---|
| 6   | 7.00   | 0.0237    | 0.166 |
| 10  | 1.00   | 0.0062    | 0.0062 |
| 20  | 0.17   | 0.0013    | 0.00022 |
| 50  | 0.023  | 0.00018   | 4.3e-6 |
| 100 | ~0     | 0.00004   | ~0 |

Density alone drops by `~600x` from `r = 6` to `r = 70` — still enough to absorb a passing sightline
meaningfully all the way out to large `r`. The emission weight drops by `~40,000x` over the same range,
because `illumination(r) ~ dilution_factor(r, R_corona) ~ (R_corona/r)^2` for `r >> R_corona` — an *extra*
`1/r^2` on top of density's own falloff. So emission is effectively confined to a thin shell near the
launch radius `R0`, while absorption draws on wind material across the whole `R0`-to-`R_out` column.

Why the flat-space P-Cygni test doesn't show this imbalance: there, the star's radius *equals* the wind's
launch radius (`R_star = R0`, by design), so illumination and density fall off on the *same* characteristic
scale and stay comparably prominent. Here, `R_corona = 3` is a compact source sitting well inside the
wind's own launch radius `R0 = 5`, illuminating a wind extending `20x` further out (`R_out = 100`) — a
small, deeply-embedded corona lighting up a much more extended wind, a real (if extreme, for these
particular numbers) astrophysical geometry, not a code defect. The lever to change this balance is
`R_corona`/`R0`/`R_out`, not a code fix.

## 5.13  How absorption and emission are normalised together

Asked whether the absorption and emission normalisations are actually linked, and whether the relative
amount of each at a given point reflects real physics, rather than being two independently-tunable knobs.
Checked both by reading `accumulate_step` and by running it directly on controlled inputs.

**The link, in the code**: both the optical-depth increment and the (pre-`exp(-tau)`) emission increment
share the *same* prefactor `alpha(E_loc) = density * kappa0 * profile(E_loc - rest_energy)` — the standard
`j = alpha * S` source-function relation, `S` being the `source` parameter. The only things that differ
are `S` itself and the `exp(-absorption[j])` self-shielding factor.

**Verified, not just read off the formula**: calling `accumulate_step` directly with
`density = 2, source = 7, g = 1.3` (arbitrary) and zero accumulated optical depth (optically thin) gives,
at every energy bin, `d_emission / d_tau = 3.18616... = source / g^3` exactly — independent of `density`,
`kappa0`, and the line profile's value at that bin, since they appear identically in both increments and
cancel in the ratio. So the *relative* normalisation between emission and absorption at any point is
exactly `S(r) / g^3`, by construction, not two independently-chosen numbers that happen to coexist.

**What this represents physically, and where "correct physics" stops**: `S(r) = corona.illumination(r)`
is the standard single-scattering source function for a conservative (non-destructive) scatterer — every
photon removed from the passing beam at rate `alpha` is assumed re-scattered isotropically, with the same
cross-section determining both processes, exactly the construction the flat-space P-Cygni case already
uses and that construction is correctly, consistently applied here too. Three real approximations sit on
top of that correct local linkage, none of them bugs, all already flagged elsewhere in this doc:
1. **Single scattering only** (`S` uses just the corona's direct light, not any previously-scattered
   photons) — fine while the wind stays modestly thin, breaks down for a very optically thick wind.
2. **No self-shielding of the illuminating beam** (`illumination(r)` assumes the corona's light reaches
   `r` unattenuated) — can overestimate `S(r)` behind optically thick wind material (Sec 6 open item).
3. **No exact photon-number balance for one sightline** — even with the `alpha`-link done correctly, a
   single observer's spectrum need not show the absorption trough's area exactly matching the emission
   bump's area; that only closes when integrating over the full 4*pi sr of "virtual observers," not one
   line of sight (Sec 4/5.1's photon-conservation investigation). A deeper trough than bump, or vice
   versa, is geometry, not a violated conservation law.

The Sec 5.11 dilution-factor shortcut (no lensing) sits underneath all of this too — it can shift `S(r)`'s
absolute scale, but does not touch the `alpha`-sharing that links emission and absorption together.

## 5.14  A `v_inf = 0.3c` run, and why the resulting line is redshifted, not blueshifted

Re-ran `ray_transfer_disc_wind` with `v_inf = 0.3` (was `0.05`), widening the energy grid to `1.0-10.0`
keV to catch the correspondingly wider range of Doppler shifts, everything else unchanged. Result: the
same qualitative shape (a deep absorption trough plus a small emission excess) but much broader
(`~4.3-8.6` keV vs. `~4.4-6.8` keV), with a comparable trough depth (`~0.20x` vs. `~0.15x` before) — the
same absorbing column now spread across a wider energy range rather than concentrated. No numerical
issues (checked for NaN/Inf; none found) despite the much larger velocity.

Asked why the absorption feature is redshifted at all, given that a sightline to the corona looks into
wind flowing *towards* the observer along it — classically that should give a *blueshifted* trough
(P-Cygni). Checked by decomposing the redshift factor `g` at every radius along one actual corona-hitting
ray into a pure-gravity piece and the wind's own kinematic contribution, rather than assuming either
dominates:
- `g_grav(r)`: the same static-observer construction used everywhere else in this doc (Sec 5.4, 5.11),
  evaluated with the wind's own velocity zeroed out — isolates gravity alone.
- `g_full(r)`: the actual value `accumulate_step` uses, with the wind's real velocity.
- `g_full/g_grav`: the wind's own kinematic contribution, factored out.

| `r` | wind speed | `g_grav` | `g_full` | kinematic factor |
|---|---|---|---|---|
| 100.0 | 0.286 | 1.009 | 0.753 | 0.746 |
| 30.2  | 0.253 | 1.034 | 0.802 | 0.776 |
| 15.6  | 0.209 | 1.070 | 0.876 | 0.819 |
| 9.57  | 0.151 | 1.125 | 0.985 | 0.876 |
| 9.06  | 0.143 | 1.133 | 1.002 | 0.884 |
| 6.54  | 0.082 | 1.201 | 1.129 | 0.940 |
| 5.04  | 0.017 | 1.287 | 1.273 | 0.989 |

The kinematic factor is `< 1` at *every* radius — the outflowing wind is indeed blueshifting things
relative to pure gravity, exactly as classical P-Cygni intuition expects; nothing backwards there. But
`g_grav` grows faster than the kinematic factor can cancel it as `r` decreases, so `g_full` crosses `1`
around `r ~ 9.3` for this ray: outside that radius the net effect is blueshift (`g_full < 1`), inside it
gravity wins and the net effect is redshift (`g_full > 1`).

**Which regime wins the overall spectrum comes down to where the material actually sits, not which effect
is individually larger.** From Sec 5.12, density falls only `~600x` from `r=6` to `r=70` — plenty of
absorbing column still sits well inside the `r~9.3` crossover — while illumination-weighted emission is
even more concentrated near `R0=5`, deep in the `g_full > 1` regime. So both the bulk of the absorbing
column and essentially all of the emission come from the gravity-dominated region, which is why the
observed feature comes out redshifted overall despite the wind genuinely blueshifting the local physics
relative to gravity alone everywhere along the line of sight. A less centrally-concentrated wind (e.g.
smaller `R0`/`R_out` ratio, or a flatter density profile) would let the outer, kinematically-blueshifted
region contribute more, pulling the net feature back towards (or past) the rest energy.

## 5.15  Moving the wind out to `R0 = 50`: a narrow, saturated trough instead of a broad, moderate one

Re-ran `ray_transfer_disc_wind` with `R0 = 50` (was `5`) and `R_out = 1000` (was `100`, keeping the same
20x `R_out/R0` ratio); everything else — `v_inf`, `v0_vinf`, `kappa0`, `R_corona`, `I_corona`,
`source_mode = illumination`, energy grid — unchanged. Result was qualitatively different from every
previous run: instead of a broad trough spanning several keV with a moderate minimum (`res ~ 0.10-0.20`
at `R0 = 5`), the spectrum sits exactly on the continuum (`res = 1.0000`) for almost the entire grid, then
drops to near-total saturation (`res` as low as `~1e-4`, i.e. the absolute flux itself collapsing near
zero, not just the ratio) in just 3-4 bins around `E ~ 6.2-6.8` keV, before snapping back to the continuum
by the next bin. `dat/ray_transfer_disc_wind_R0_compare.png` shows the two spectra side by side.

Checked this isn't a resolution artifact by measuring the peak optical depth directly (a diagnostic scan
of `RayTransfer::trace_pixel`'s `absorption` output over the full image-plane grid, same technique as the
Sec 5.5 `R0` scan):

| `R0` | `R_out` | peak `tau` |
|---|---|---|
| 5  | 100  | 279 |
| 50 | 1000 | 540 |

*(Correction: an earlier version of this table, from a first diagnostic script (`/tmp/r050_check.cpp`),
reported `38.3` and `249.4` — that script had `if (!rt.trace_pixel(...)) continue;`, which skipped every
pixel whose ray did *not* terminate on the corona before updating the running max. Since only a small
minority of pixels ever hit the corona (`541/10000` in this geometry) and the pixel with the true peak
column is not generally one of them, that silently discarded most of the grid, including the actual
worst-case sightline, and underestimated both numbers. Re-checked with a corrected script that processes
`absorption` unconditionally (matching the style already used in Sec 5.16's diagnostics, which were never
gated this way and are unaffected) — the table above is the corrected version; the bug was in this one
diagnostic script, not in `RayTransfer`/`accumulate_step` itself, and the corrected `R0 = 5` peak (`279`)
also matches Sec 5.16's own independent measurement of the same configuration.)*

Consistent in direction with Sec 5.5's finding that near-launch optical depth pileup grows with `R0`, but
the growth is much milder than first reported: a 10x increase in `R0` (at fixed `R_out/R0`) drove peak
`tau` up by only `~1.9x` (not the `~6.5x` originally claimed). At either value `exp(-tau)` is already
negligible at the peak, so both configurations are optically thick there — the *depth* difference alone
doesn't explain why `R0 = 50`'s trough is so much narrower.

The narrowing comes from where that column sits relative to how fast the redshift changes with radius.
At `R0 = 5`, the wind sits close to the black hole, where both the gravitational redshift (Sec 5.14's
`g_grav(r)`) and the velocity law `v(r)` vary quickly with `r` — the absorbing column near the launch
shell is spread over a range of radii with a correspondingly wide range of redshifts, so the (smaller)
total optical depth gets smeared across many energy bins, giving a broad, moderate trough. At `R0 = 50`,
gravity is weak and slowly varying, and `v(r)` sits close to its base value `v0` over a wide radial range
near the (now much larger) launch shell — so a large fraction of the much bigger absorbing column shares
nearly the same redshift. Instead of being smeared out, it piles into just a few energy bins, producing a
narrow, deeply saturated notch surrounded by essentially unmodified continuum. Genuine consequence of the
gravitational-redshift gradient flattening out at large `r`, not a bug or a numerical artifact.

### Why there's no blueshifted emission/absorption on the near side of the line

**Terminology check first (correcting an error in earlier session notes for this sub-topic): `g > 1` is
redshifted, `g < 1` is blueshifted (Sec 5.9's sign-fix discussion), so a point with `g(r) > 1` resonates at
an *observed* energy `E_obs = rest_energy / g < 6.4` keV — i.e. `E < 6.4` keV is the redshifted side and
`E > 6.4` keV is the blueshifted side, matching Sec 5.14's own language ("classically that should give a
blueshifted trough" for the near/approaching side) and the original Sec 4 test ("blueshifted absorption
trough (residual `< 0.9` *above* the line energy)"). An early pass at this sub-topic had these two swapped;
the physical mechanism below is unaffected, but the labels are corrected throughout.**

Asked why we don't see the near (approaching) side of the wind show up as a genuinely *blueshifted*
absorption/emission feature (`E > 6.4` keV), the way a classical, gravity-free P-Cygni profile would show
it. With correct labels, this is *not* "the far/receding side is missing" — the far side's redshifted
contribution (`E < 6.4` keV) is exactly where both the R0 = 5 and R0 = 50 spectra peak. What's actually
almost entirely absent is the blueshifted side, `E > 6.4` keV.

The mechanism is Sec 5.14's gravity-vs-kinematics competition, now seen at the level of a single ray: on
the near/ingoing leg, `g` sweeps from `< 1` (blueshifting, far from the black hole, kinematics winning) up
through `1` and on to `> 1` (redshifting, near periapsis, gravity winning) — Sec 5.14's crossover radius.
For `R0 = 5`, most of the wind's density (and hence most of the absorbing/emitting column) sits *inside*
that crossover, deep in the `g > 1`/redshifted regime, so only a small, low-density fraction of the near
leg (the part still far enough out for kinematics to win) ever produces true blueshift — that weak
contribution is *also* encountered first along the backward-traced ray, before much column has built up
(`absorption[j]` starts at `0`, `ray_transfer.cpp:252-253`), so it isn't itself heavily self-shielded; it is
just intrinsically weak because there is little density out there to begin with (Sec 5.12). The far/outgoing
leg, by contrast, has gravitational and kinematic (receding) redshift *reinforcing* each other the whole
way (`g` never drops much below `~1`), so it contributes only to the already-dominant redshifted side, and
whatever it does contribute is additionally suppressed by self-shielding from the near-side column ahead of
it (same `exp(-absorption[j])` mechanism, `accumulate_step`, `ray_transfer.cpp:120-146`) — a real one-ray
correct-physics effect, but not the reason the blueshifted side specifically is weak.

So the blueshifted side is suppressed by *low density* at the only radii where kinematics can win (far from
the black hole, where gravity doesn't override it), not primarily by self-shielding. This predicts a clean
test: push `R0` out far enough that kinematics wins over gravity at (and even below) the wind's own peak
density radius — taken up directly below, per the follow-up question, in "Recovering the classical P-Cygni
shape at large R0."

### Verifying the resonance condition itself: local absorber frame, not the emitter's

Asked to double-check the underlying logic: resonant-line absorption should only happen to a photon when
its energy *in the local rest frame of the absorbing material* matches the line, not when some fixed
"emitter's redshift" matches anything. Checked this is what the code actually does, both by reading it and
by tracing one ray's `g` in detail.

In `trace_pixel`'s per-step loop (`ray_transfer.cpp:249-330`), `g` is recomputed from scratch at *every*
step — it is not tied to the corona's redshift, or to any other point's redshift, or to any global/fixed
reference. Each step reconstructs the photon's exact 4-momentum at the *current* `(r, theta, phi)` from
the ray's conserved constants of motion `(k, h, Q)` (`momentum_from_consts`, an exact reconstruction, not
an approximation — the same one every propagator in this codebase uses), then dots it against the wind's
own local 4-velocity *at that same point* (`m_field.four_velocity(r, q.theta, q.phi, ...)`, also evaluated
fresh each step, via `Raytracer::ray_redshift(const T et[4], ...)`, `raytracer.cpp:574-618`). `E_loc =
bins.energy[j] * g` is therefore exactly "what a given observed spectral channel's photon looks like,
locally, to whatever material actually sits at this point" — `accumulate_step` then gates both absorption
and emission on `line.profile(E_loc - line.rest_energy)`, i.e. on *that point's own* redshift relative to
the observer. There is no step at which the corona's (or any other point's) redshift is substituted in.

Confirmed this numerically by instrumenting one ray directly (R0 = 5 baseline, a pixel with impact
parameter `~11.3` grazing periapsis at `r ~ 10` without diving inside `R0`, so it samples the wind on both
the near/ingoing and far/outgoing legs) and printing `g` and the corresponding resonant observed energy
(`rest_energy / g`) along the whole path:

| leg | `r` | `g` | resonant `E_obs = 6.4/g` | side |
|---|---|---|---|---|
| near (ingoing) | 100 -> 10.1 | 0.963 -> 1.146 | 6.65 -> 5.59 | blueshifted -> redshifted |
| far (outgoing) | 10.1 -> 94.4 | 1.146 -> 1.059 | 5.59 -> 6.05 | redshifted throughout |

`g` varies continuously and *differently* between the two legs at the same radius (e.g. at `r ~ 94-100`:
ingoing `g = 0.963`, outgoing `g = 1.059`) — exactly as expected, since the wind's local velocity relative
to the photon's actual momentum direction there differs between the two passes; nothing here is reusing a
single fixed value. This confirms the per-point resonance logic is genuinely local, not a shortcut, and
directly shows the near leg crossing from blueshifted to redshifted while the far leg stays redshifted
throughout — consistent with Sec 5.14 and with the paragraph above.

## 5.16  Recovering the classical P-Cygni shape at large R0

Hypothesis to test: if `R0` is pushed far enough out that gravity is both weak *and* slowly-varying
compared to the wind's own kinematic Doppler shift, and the field of view is wide enough that most rays
don't have to dive close to the black hole to reach the wind at all, the near/far asymmetry above should
relax and something closer to the classical, gravity-free P-Cygni shape (a blueshifted trough, a genuinely
separate redshifted emission excess) should re-emerge — exactly the physical picture behind Sec 5.14's
crossover radius, taken to its limit `R0 -> infinity`.

Tested with a lightweight diagnostic (`RayTransfer::trace_pixel` scanned directly over the image-plane
grid, bypassing the full FITS/CSV application, same technique as Sec 5.15's checks) summing line emission
into "redshifted" (`E < 6.4` keV) and "blueshifted" (`E > 6.4` keV) totals, and tracking peak `tau`, for
several `(R0, R_out, field-of-view)` combinations (`v_inf = 0.05`, `v0_vinf = 0.05`, `n0 = 1`, `R_corona =
3`, `I_corona = 1` throughout — only geometry and `kappa0` varied):

| `R0` | `R_out` | fov | `kappa0` | redshifted | blueshifted | blue/red | peak `tau` |
|---|---|---|---|---|---|---|---|
| 5    | 100   | 20   | 0.2    | 620.5 | 25.9  | 4.2%  | 275 |
| 200  | 2000  | 800  | 0.2    | 0.280 | 0.060 | 21.4% | 9598 |
| 200  | 2000  | 2000 | 0.2    | 0.064 | 0.023 | 35.1% | 7987 |
| 1000 | 10000 | 4000 | 0.2    | 0.0114| 0.0021| 18.2% | 48533 |
| 200  | 2000  | 2000 | 0.002  | 0.037 | 0.023 | 63.2% | 79.9 |
| 200  | 2000  | 2000 | 0.0002 | 0.0143| 0.0097| 67.7% | 7.99 |
| 5    | 100   | 20   | 0.0002 | 6.214 | 0.052 | 0.83% | 0.27 |

Three separable effects, disentangled by varying one thing at a time:

1. **Weaker/slower-varying gravity at large `R0` does help, but only partially on its own.** Going from
   `R0 = 5` to `R0 = 200` (fov/R0 held at `4` in both) took the blue/red ratio from `4.2%` to `21.4%` — a
   real improvement, but far short of the classical, order-unity balance, and pushing `R0` further (to
   `1000`, same fov/R0 ratio) did *not* continue improving it (`18.2%`, if anything slightly worse) —
   because peak `tau` also kept climbing (`9598 -> 48533`), consistent with Sec 5.15's finding that
   near-launch pileup grows with `R0`, and self-shielding (previous section) eating into the gain.
2. **A wider field of view *relative to R0* helps on top of that.** Holding `R0 = 200` fixed and widening
   fov/R0 from `4` to `10` took blue/red from `21.4%` to `35.1%` — consistent with more of the grid's solid
   angle sampling gentle, non-plunging paths through the low-density outer wind rather than grazing the
   dense launch shell.
3. **Neither of those alone breaks the self-shielding ceiling; reducing the optical depth does.** At
   `R0 = 200`, fov `= 2000`, the original `kappa0 = 0.2` gives `tau` up to `~8000` and a lopsided `35%`
   ratio; dropping `kappa0` by `100x` (`0.002`) collapses `tau` to `~80` and the ratio jumps to `63%` —
   comparable magnitudes, much closer to the classical picture — and a further `10x` drop (`kappa0 =
   0.0002`, `tau ~ 8`) only nudges it further (`68%`), i.e. it has converged once genuinely optically thin.
4. **Weak gravity is still necessary, not just optical thinness.** Re-running the *original* `R0 = 5`
   geometry at the same very low `kappa0 = 0.0002` (`tau ~ 0.27`, negligible self-shielding) gives
   blue/red `= 0.83%` — *worse* than the thick `R0 = 5` case, if anything, and nowhere near the `R0 =
   200` result at the same `kappa0`. So the small-`R0` asymmetry is not an optical-depth artifact at all:
   even with self-shielding switched off, gravity still dominates the kinematics across essentially the
   whole `R0 = 5` wind (Sec 5.14's crossover sits at `r ~ 9.3`, deep inside most of the density), so the
   blueshifted side stays intrinsically starved of material regardless of `tau`.

So the hypothesis is confirmed, with a caveat the field-of-view argument alone doesn't capture: recovering
the classical P-Cygni shape needs *all three* — `R0` large enough that gravity is weak where the density
peaks, a field of view wide enough to sample non-plunging geometry, and `kappa0`/density low enough that
the (otherwise ever-growing, Sec 5.15) near-launch column doesn't self-shield the far side into invisibility.
Pushing `R0` out on its own, at fixed `kappa0`, makes the self-shielding problem *worse* faster than it
fixes the gravity problem — a genuine tension between the two knobs the naive "just make R0 bigger" picture
misses. Not run as a full disc_wind/FITS application (this was a scan of aggregate emission sums via the
lightweight diagnostic only); a production-quality large-`R0` P-Cygni comparison would need a proper run at
a `kappa0` tuned into the optically-thin-enough regime for that `R0`, plus convergence checks (Sec 6).

## 5.17  Per-line-of-sight optical depth as a built-in diagnostic

Sec 5.15/5.16 needed the peak wind optical depth along a line of sight (thick vs thin) often enough that
it's now a permanent output of both applications, rather than a one-off script each time:

- `ray_transfer_disc_wind.cpp`: a new `TAU` FITS extension, `tau_map[ix][iy] = max_j absorption[j]` per
  pixel — `absorption[j]` is already accumulated over the *entire* backward-traced ray and never reset
  (`trace_pixel`, `ray_transfer.cpp:252-253`), so this genuinely is the total wind optical depth along
  that whole line of sight, not a per-step or per-leg value. Also prints a one-line summary (peak, mean,
  and the fraction of sightlines with `tau > 1`) after the main loop.
- `ray_transfer_kerr_vs_flat.cpp`: the same peak/mean/fraction-thick summary printed to stdout for both
  the Kerr and flat sides (no FITS output there to attach a map to).

While adding this, re-checked Sec 5.15's peak-`tau` table against it and found it had been *understated*:
the very first diagnostic script used to produce that table (`/tmp/r050_check.cpp`, since deleted) had
`if (!rt.trace_pixel(...)) continue;`, silently skipping every pixel whose ray didn't terminate on the
corona — the vast majority of pixels, and not generally the one with the true peak column — before
comparing against the running maximum. The bug was confined to that one script; `RayTransfer` itself, and
every other diagnostic used from Sec 5.16 onward (which process `absorption` unconditionally), were not
affected. Sec 5.15's table has been corrected (`R0 = 5`: `38.3 -> 279`; `R0 = 50`: `249.4 -> 540`) and its
"`~6.5x` growth" claim corrected to "`~1.9x`"; the qualitative conclusions there (both configurations
saturate `exp(-tau)`, the narrowing comes from redshift concentration not from the depth difference) are
unaffected by the correction, and the new built-in `TAU` output is now the reference for these numbers
going forward rather than a fresh one-off script each time.

## 5.18  A full `R0 = 200` run at peak wind tau = 1

Requested: an actual `ray_transfer_disc_wind` production run (not just the lightweight diagnostic used in
Sec 5.16) at `R0 = 200`, with `kappa0` tuned so the peak wind optical depth is exactly `1` — the
thick/thin boundary.

**A real resolution limit showed up first.** `tau_corona_max` (Sec 5.17) requires a pixel to actually
terminate on the corona; at a field of view wide enough to matter for the Sec 5.16 exploration (`x0/xmax =
-800/800`, i.e. `fov/R0 = 4`, the same ratio used throughout this investigation) and the usual `Nx = Ny =
100`, **zero pixels hit the corona** — confirmed at two grids (`40x40` and `100x100`). The corona's
capture cross-section on the image plane is set by `R_corona = 3` and the BH/lensing geometry alone,
independent of `R0`/`R_out`/field of view; calibrated against the `R0 = 5`, `fov = 20` baseline (`~300-540`
hits out of `10000`, Sec 5.15/5.17), the expected hit count scales as `~12 / fov^2` pixels for a unit-area
grid — at `fov = 800` that predicts `~0.2` expected hits out of `10000`, consistent with the `0` actually
seen, and getting `O(10)` hits at this `fov` would need a grid of order `1000 x 1000`, ~100x this run's
cost. Used `tau_map`'s *global* peak (Sec 5.17) as the practical stand-in for `tau_corona_max` instead,
reasoning that any ray reaching the corona has to cross the same near-launch column that sets the global
peak in the first place, so the two should be comparable in scale even though the global peak isn't
literally restricted to corona-hitting rays.

Tuned by linearity (`tau` is exactly proportional to `kappa0`, all else fixed, since `d_tau = density *
kappa0 * profile * dl` — verified by the tuning landing exactly on `tau = 1.0`, not approximately):
measured global peak `tau = 95.83` at `kappa0 = 0.002` (`fov = 800`, `R0 = 200`, `R_out = 4000`, `dist =
6000`, `Nx = Ny = 100`, otherwise the usual `v_inf = 0.05`, `v0_vinf = 0.05`, `R_corona = 3`, `I_corona =
1`), then scaled `kappa0 -> kappa0 * (1/95.83) = 2.087e-5` and re-ran: `peak tau = 1` exactly, `0/10000`
sightlines thick (as expected right at the threshold), still `0/10000` corona hits (a resolution property,
independent of `kappa0`).

**Result:** `dat/ray_transfer_disc_wind_R200_tau1_spectrum.png` — a single, smooth, roughly
Doppler-shaped emission line, peak at `E ~ 6.366` keV (a small residual redshift from the still-nonzero,
if now weak, gravity at `r ~ R0 = 200`), width comparable to the kinematic `v_inf` scale, falling off
smoothly to numerically zero by `~5.85`/`~6.95` keV with no secondary features. Slightly asymmetric
(broader on the blueshifted side of its own peak than the redshifted side) rather than perfectly
symmetric, but recognisably a single classical-looking Doppler line, not the earlier saturated/narrow
notch (Sec 5.15) or the heavily lopsided red-dominated shape (Sec 5.15-5.16) — consistent with Sec 5.16's
finding that `R0 = 200` plus a reasonably wide field of view plus low-enough `kappa0` moves the spectrum
back towards the classical picture.

**Important caveat, a direct consequence of the `0` corona hits above: there is no absorption trough to
see in this result.** `continuum = 0` for every pixel, so `total_flux == line_flux` exactly and the CSV's
`residual` column (`total_flux / continuum_total`) is `0` everywhere (division guarded against
`continuum_total = 0`) — not because there's no absorption, but because no sightline in this run samples
the continuum at all. The classic P-Cygni "dip below the continuum" specifically needs rays that terminate
on the corona; what this run actually shows is the wind's own line emission in isolation, self-consistently
including whatever self-shielding already happened along each ray (Sec 5.15's mechanism), at the
optically-thin/thick boundary. A true P-Cygni-shaped comparison (trough *and* bump together) at this `R0`
would need either a far larger pixel grid (as estimated above) or a narrower field of view sacrificing the
"avoid diving into the cavity" property that motivated the wide `fov` in the first place — the same
tension identified in Sec 5.16 between field-of-view width and corona-hit statistics, now hit directly
rather than just inferred from aggregate sums.

## 5.19  `R0 = 50` at peak wind tau = 1 -- an actual absorption trough, and the global/corona-tau gap

Repeated Sec 5.18's exercise at `R0 = 50` (`R_out = 1000`, same `20x` ratio; `dist = 1500`; field of view
scaled the same way, `fov/R0 = 4` -> `x0/xmax = -200/200`; `Nx = Ny = 100`; energy grid widened slightly to
`5.4-7.3` for the somewhat stronger gravity at this smaller `R0`; otherwise the usual `v_inf = 0.05`,
`v0_vinf = 0.05`, `R_corona = 3`, `I_corona = 1`).

At this (smaller, in absolute terms) field of view, `2 / 10000` pixels actually hit the corona -- enough
to measure `tau_corona_max` directly rather than substituting the global peak (Sec 5.18's `fov = 800`
gave `0` hits; this `fov = 200` predicts `~3` expected hits by Sec 5.18's `~12/fov^2` scaling, consistent
with the `2` seen). Measured `tau_corona_max = 2.744` at `kappa0 = 0.002`, scaled by the same exact
linearity as before to `kappa0 = 7.289e-4`, re-ran: `tau_corona_max = 1` exactly.

**Result:** `dat/ray_transfer_disc_wind_R050_tau1_spectrum.png` -- a real P-Cygni shape this time, not just
an isolated emission line: a modest emission excess (`residual` peaking at `~1.066` around `E ~ 6.19` keV)
immediately followed by a genuine absorption dip *below* the continuum (`residual` down to `~0.593` around
`E ~ 6.34` keV), smoothly returning to `1` on both wings. Unlike Sec 5.18, this run actually has `continuum
> 0` pixels to show the trough against, so both halves of the classic P-Cygni shape are visible together
for the first time in this `R0`-scan sequence.

**The global and corona-specific peak `tau` disagree substantially here: `9.246` vs `1` (a factor `~9x`).**
This directly tests Sec 5.18's assumption (there forced, for lack of any corona hits, to use the global
`tau_map` peak as a stand-in for `tau_corona_max`) and shows it is not a reliable substitute in general --
the sightline with the single deepest column overall need not be, and here is not, one that happens to
land exactly on the corona. Practically, this means Sec 5.18's `R0 = 200` run was tuned to global peak
`tau = 1`, but its *true* `tau_corona_max` (unmeasurable there at that resolution) could plausibly be well
below `1` -- i.e. that run may have been over-attenuated relative to what "peak optical depth to the
corona = 1" actually calls for. Worth re-checking with a higher-resolution or narrower-`fov` `R0 = 200` run
if an accurate `R0 = 200` P-Cygni comparison at exactly `tau_corona_max = 1` is wanted (Sec 6).

## 5.20  `R0 = 50` at peak wind tau = 0.5

Same geometry as Sec 5.19 (`R0 = 50`, `R_out = 1000`, `fov = 200`, `dist = 1500`, `Nx = Ny = 100`), halving
`tau_corona_max` to `0.5` by the same exact linear scaling (`kappa0: 7.289e-4 -> 3.645e-4`) -- confirmed by
the run: `tau_corona_max = 0.5` exactly, global peak `tau = 4.623` (also almost exactly half of Sec 5.19's
`9.246`, as expected since nothing else changed).

**Result:** `dat/ray_transfer_disc_wind_R050_tau_compare.png` overlays the two directly. Both the emission
bump and the absorption dip shrink towards the continuum, roughly proportionately: the bump's peak
`residual` drops from `1.066` (Sec 5.19) to `1.043`, and the dip's minimum rises from `0.593` to `0.761`
(so the dip's *depth below continuum*, `1 - residual`, goes from `0.407` to `0.239` -- a factor `~1.7`,
not exactly `2x`, consistent with the trough not being a simple `1 - exp(-tau)` single-column relation
once source-function/emission contributions and the spread of `g` over many contributing radii are folded
in, but moving in the expected direction and order of magnitude as `tau_corona_max` is halved). Same
overall P-Cygni shape and energies in both cases -- halving the optical depth changes the strength of the
feature, not its qualitative form or location, as expected for two runs that only differ in `kappa0`.

## 5.21  `R0 = 50` at peak wind tau = 2

Same geometry again, `kappa0: 7.289e-4 -> 1.458e-3` (`2x` Sec 5.19's value) -- confirmed: `tau_corona_max =
2` exactly, global peak `tau = 18.49` (also `2x` Sec 5.19's `9.246`).

`dat/ray_transfer_disc_wind_R050_tau_compare.png` now overlays all three (`tau_corona_max = 0.5, 1, 2`).
Trough minimum `residual`: `0.761 -> 0.593 -> 0.424` (depth below continuum `0.239 -> 0.407 -> 0.576`);
successive depth ratios `1.70x` then `1.41x` for the same `2x` step in `tau_corona_max` each time --
monotonically deepening but with clearly diminishing returns as it approaches saturation (depth `-> 1`),
not a linear `depth ~ tau` relation, consistent with a trough shaped by `1 - exp(-tau)`-like attenuation
of the continuum rather than a pure optically-thin scaling. Same energies and overall shape throughout
(emission bump `~6.17-6.19` keV, trough minimum fixed at `~6.34` keV) -- only the strength of the feature
tracks `kappa0`/`tau_corona_max`, as in Sec 5.20.

## 5.22  `ConicalBetaWind`: a wind restricted to a range of polar angle

Added a new `RTField` (`ray_transfer.h`), for winds launched from a limited latitude range rather than
filling the whole sphere -- two configurations requested: hugging the disc surface, hollow near the poles;
or collimated along the polar axis, hollow near the equator.

`WindLatitudeMode { DiscLaunched, PolarCollimated }` plus a single `theta_lim` (measured from the pole,
`theta = 0`, matching the Boyer-Lindquist convention used throughout `kerr.h`/`raytracer.h` where `theta =
pi/2` is the equatorial plane) selects between them, both symmetric about the equator:
- `DiscLaunched`: wind fills `theta_lim < theta < pi - theta_lim` -- absent near either pole.
- `PolarCollimated`: wind fills `theta < theta_lim || theta > pi - theta_lim` -- absent near the equator.

`ConicalBetaWind<T>` implements this by composition, not by duplicating or re-deriving any physics: it
holds a `SphericalBetaWind<T> radial` member and delegates `density()`/`four_velocity()`/
`flat_four_velocity()` straight through unchanged (the beta-law velocity/density profile never depended
on `theta` to begin with -- only whether a given `theta` is *included* is new), and `in_wind()` becomes
`radial.in_wind(r, theta, phi) && in_theta_range(theta)`. So a `ConicalBetaWind` and a `SphericalBetaWind`
built with the same `(v_inf, v0, beta_exp, R0, R_out, n0)` are physically identical everywhere both
consider themselves "in the wind" -- confirmed by a direct check (density and all four `four_velocity`
components bit-identical, `diff = 0`, at several `(r, theta)` inside both regions).

Verified the latitude gating itself with a scan of `theta` from pole to pole at fixed `theta_lim = 30 deg`:
`DiscLaunched` and `PolarCollimated` are exact complements of each other (never both `true`/both `false`
at the same `theta`) and both are correctly symmetric about the equator (`theta` and `pi - theta` give the
same `in_wind` verdict); the radial bounds (`r <= R0` or `r >= R_out`) still exclude a point regardless of
which latitude mode or `theta` it's at.

No application built on this yet -- `ray_transfer_disc_wind.cpp`/`ray_transfer_kerr_vs_flat.cpp` still
construct a plain `SphericalBetaWind`; wiring `WindLatitudeMode`/`theta_lim` through their par files (and
choosing what happens when `theta_lim` isn't set, e.g. defaulting to the existing full-sphere behaviour)
is the natural next step once specific disc-launched/polar-collimated science cases are ready to run.

## 5.23  Wiring `ConicalBetaWind` into `ray_transfer_disc_wind` and comparing all three geometries

Wired Sec 5.22's `ConicalBetaWind` into the application: a new `wind_geometry` par ("spherical", the
default; "disc"; "polar") plus `theta_lim` (degrees, required for "disc"/"polar"). Implemented with a
`unique_ptr<RTField<double>>` chosen at runtime (`RayTransfer`'s constructor already takes `const
RTField<T>&`, so this needed no interface change) rather than three near-duplicate code paths.

Ran the `disc` and `polar` geometries at `theta_lim = 30` deg, otherwise *identical* to Sec 5.19's `R0 =
50`, `tau_corona_max = 1` spherical run (same `kappa0`, same everything) -- so the comparison isolates the
effect of the wind's latitude range alone, not a re-tuned optical depth for each:

| geometry | corona hits | `tau_corona_max` | global peak `tau` | emission peak `residual` | trough min `residual` |
|---|---|---|---|---|---|
| spherical | 2 | 1 | 9.246 | 1.066 | 0.593 |
| disc (`theta_lim=30`) | 2 | 1 | 9.02 | 1.050 | 0.572 |
| polar (`theta_lim=30`) | 2 | 0 | 4.367 | 1.058 | 1.000 (no trough) |

`dat/ray_transfer_disc_wind_R050_geometry_compare.png` overlays all three.

**Disc-launched nearly reproduces the spherical result.** At `incl = 60` deg, the observer's line of
sight sits comfortably inside the `30`-`150` deg band the disc wind fills, and evidently so does essentially
every ray that contributes to the peak column -- `tau_corona_max` is identical (`1`, unsurprising since the
two corona-hitting rays apparently never left the `30`-`150` band even in the spherical run) and the global
peak drops only `~2.4%`. Restricting the wind to the disc region barely matters when nothing you're
looking at (at this inclination) needed the polar caps anyway.

**Polar-collimated removes the absorption trough entirely.** `tau_corona_max = 0` exactly: at `incl = 60`
deg, the direct line of sight to the corona never dips into either polar cone (`theta < 30` or `> 150`),
so none of the wind's column lies along that specific sightline and the continuum reaches the observer
completely unabsorbed -- `residual` never drops below `1` anywhere in the spectrum. What line emission
there is (peak `residual = 1.058`, weaker than the spherical/disc case's `line_flux` peak by `~4x`) comes
entirely from *other* pixels whose rays do cross the polar cones and are illuminated by the corona, with
no accompanying absorption feature at all -- a pure emission line sitting on an unabsorbed continuum,
qualitatively different from the P-Cygni shape the spherical/disc geometries give at this same
inclination. (A more face-on/polar inclination would flip this: the direct sightline would then thread the
polar cone instead, restoring an absorption trough for the polar-collimated case and likely removing it
for the disc-launched one -- not tested here, but a direct, cheap follow-up given the machinery now exists.)
