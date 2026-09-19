# `CausticBundle` — class-based bundle tracer for `caustic_3d` and `caustic_ent`

Plan (2026-09-18). Status: **planned, not yet implemented.**

## 1  Goal

Replace the free functions and loose structs of `src/caustic/caustic_bundle.h` (`BundleRay`, `BundleParams`,
`bundle_from_rays`, `bundle_step`, `bundle_jacobian`, `rescale_pair`, `trace_bundle`, `disc_crossing_redshift`) and
the `SingleRayBundle` helper by one class, `CausticBundle<T>`, that owns everything belonging to one bundle: its
five rays, their Mino-time steppers, the running bookkeeping (Mino time, step count, rescaling, caustic and
equatorial-crossing counts) and its results. The flow in both applications becomes: build the `ImagePlane`;
for every pixel (regular grid, adaptive refinement, contour-vertex bisection alike) construct a
`CausticBundle` from the plane and the pixel's `(x, y)`; call `trace()`; read the results off the object.
Numerical behaviour is unchanged and verified bundle-by-bundle against the current code.

## 2  Library changes (reuse rather than re-implement)

- `ImagePlane<T>::init_ray(Ray<T>& ray, T x, T y) const` (new, public): the body of the per-pixel loop of
  `init_image_plane` — position and momentum on the plane at `(x, y)`, constants of motion, signs, `alpha/beta`.
  `init_image_plane` calls it for every pixel, so every imaging application is unchanged (the `ImagePlane`
  regression in `caustic_3d_test` (c) and the other integrator tests remain the check). This removes the need
  for five shifted `ImagePlane`s in `caustic_3d`/`caustic_ent` and for `SingleRayBundle` altogether.
- `Raytracer<T>::calculate_constants_from_p` gets a `Ray<T>&`-based overload (`constants_from_p(Ray<T>&, pt,
  pr, ptheta, pphi) const`) that the existing index-based version calls; `init_ray` uses it.
- `Raytracer<T>::emit_energy(const Ray<T>&, T V, bool reverse) const` — what `redshift_start` computes per
  ray, exposed for one ray (`redshift_start` calls it in its loop). The bundle uses it for the observer-frame
  energy of its centre ray instead of interpolating `emit` between pixels.
- Everything else the bundle needs already exists and is used as is: `MinoState`, `MinoStepper::step`,
  `mino_step_size` (`mino_stepper.h`); `mino_init`, `mino_r_from_u`, `mino_phidot`, `carter_Q`, `cartesian`,
  `kerr_horizon` (`kerr.h`); `Raytracer::ray_redshift` for the disc redshift.
- Kept local to the class because there is no library equivalent and they are on the hot path: the Cartesian
  velocity `dX/dλ_M` from the Mino state (`ray_velocity`), the 3×3 determinant, the finite-difference Jacobian,
  the pair rescaling, `wrap_phi`, `crossed_equator`.

## 3  Class design (`src/caustic/caustic_bundle.h`)

```cpp
template <typename T = double>
class CausticBundle
{
public:
    struct Params            // what BundleParams holds today, with the same defaults
    {
        T h0, r_cap, max_tstep, maxtstep_rlim, max_phistep, r_max, delta, delta_max;
        int order, steplim, max_caustics;
        int max_eqcross = 0, stop_after_eqcross = 0;
        static Params from_plane(const ImagePlane<T>& plane, T delta, ...);   // fills a, horizon, r_cap
    };
    enum Status { ESCAPED, HORIZON, STEPLIM, SPLIT, SKIPPED, NAN_STATE, EQSTOP };   // = today's STATUS_* values

    // build the bundle centred on image-plane position (x, y): centre ray + (x ± δ, y), (x, y ± δ) from
    // plane.init_ray(), converted to the canonical Mino-time state with mino_init(); level/ix/iy are labels
    CausticBundle(const ImagePlane<T>& plane, const Params& P, T x, T y, int ix = 0, int iy = 0, int level = 0);

    bool valid() const;                 // initial data defined (false for the x = y = 0 ray)
    void trace(std::ostream* dump = nullptr);   // the loop of trace_bundle(): steps, caustic bisection, equatorial
                                                // crossings (if max_eqcross > 0), rescaling, termination

    // results (filled by trace())
    Status status() const;  int ncaust() const;  int eqcross() const;  T tau_end() const;
    const std::vector<CausticPoint>& caustics() const;        // as today (with level)
    const std::vector<DiscCrossing>& disc_crossings() const;  // as today
    const CausticPoint* first_caustic() const;                // r1/theta1/phi1 of the PixelResult

    // state and derived quantities of the current bundle
    T jacobian() const;                                       // J of the (rescaled) bundle now
    T log_scale() const;                                      // accumulated log10 rescaling
    const MinoState<T>& centre() const;
    T emit() const;                                           // observer-frame energy of the centre ray
    T disc_redshift(const DiscCrossing& dc) const;            // E_disc/E_obs via plane.ray_redshift(V = -1, reverse)
    T x() const; T y() const; int ix() const; int iy() const; int level() const;

private:
    struct Member { MinoState<T> s; T k, h; };                // BundleRay
    const ImagePlane<T>& m_plane;  Params m_P;  T m_a;         // m_a = propagation spin = -plane spin
    Member m_ray[5], m_prev[5], m_trial[5];  MinoStepper<T> m_st[5];
    T m_tau, m_logscale, m_J_prev;  int m_steps, m_rflips, m_eqcross;  bool m_valid;
    Status m_status;  std::vector<CausticPoint> m_caustics;  std::vector<DiscCrossing> m_disc;  int m_ncaust_total;

    int  step_all(Member r[5], T step);                       // bundle_step
    T    jacobian_of(const Member r[5]) const;                // bundle_jacobian
    void position(const Member&, T X[3]) const;  void velocity(const Member&, T V[3]) const;
    void rescale_pair(int i, int j, T s);
    template <class Pred> T bisect_step(T step, Pred crossed);   // shared by the caustic and the equatorial bisection:
                                                              // re-takes the composed step from m_prev into m_trial
    void record_caustic(T frac, T step);  void record_disc_crossing(T frac, T step, T caustic_frac);
    void check_rescale();
};
```

The two bisections (`J` sign change; `θ − π/2` sign change) currently duplicate the "re-take the step from the
saved states by bisection on the fraction" loop; `bisect_step` takes a predicate on `m_trial` so both share it
(same iteration count and tolerance as now, so the located fractions are bitwise identical).

`CausticPoint`, `DiscCrossing` and `PixelResult` stay as plain result structs (the FITS writers use them);
`PixelResult` becomes what `status()/ncaust()/eqcross()/tau_end()/first_caustic()` return, assembled by a
`result()` accessor for the per-pixel maps.

## 4  Application changes

- `caustic_3d.cpp`: one `ImagePlane` (no `planes[5]`, no `ray_arrays`), `Params::from_plane(...)`, and the pixel
  loop becomes `CausticBundle b(plane, P, plane.ray_x(pix), plane.ray_y(pix), ix, iy); if (!b.valid()) ...;
  b.trace(); results[pix] = b.result(); append b.caustics()`. The adaptive-refinement loop and the `dump_pixel`
  diagnostic use the same constructor with `(x, y)` from `AdaptivePlane`. `SingleRayBundle` goes.
- `caustic_ent.cpp`: likewise for the grid, the refinement and the vertex bisection (`eval(x, y)` becomes
  "construct, trace with `stop_after_eqcross = n`, take `disc_crossings()`"); `g` from `b.disc_redshift(dc)`,
  `emit` from `b.emit()` (exact at every point, replacing the linear interpolation between pixels in the
  bisection — a change at the 1e-6 level in `G` for refined vertices, checked in §5).
- `caustic_3d_test.cpp`, `caustic_ent_test.cpp`: use the class; the checks and tolerances are unchanged.
- `single_ray_bundle.h` deleted; `caustic_bundle.h` rewritten; `CLAUDE.md` and the two design notes updated.

## 5  Verification

1. **Bundle-level regression test** (`src/tests/caustic_bundle_regression_test.cpp`, no cfitsio): the current
   header is kept during the migration as `caustic_bundle_legacy.h` (namespace `legacy`). For `a = 0.998,
   i = 60°` and `a = 0, i = 30°`, on a `41×41` grid over `±8 rg` plus 200 random off-grid positions and with
   `max_eqcross = 3`, every bundle is traced both ways; the test requires identical `status`, `ncaust`,
   `eqcross`, `tau_end`, identical caustic lists (`n, tau, t, r, θ, φ, rflips, eqcross, dj_sign, logscale`
   bitwise equal — the arithmetic is the same and executed in the same order) and identical disc crossings
   (`J, r, φ, t, u, p_u, p_θ, ncaust`), and `disc_redshift` equal to `legacy::disc_crossing_redshift` to
   1e-14. The off-grid positions are initialised with `init_ray` for the class and with a single-ray
   `ImagePlane` for the legacy path, which also checks `init_ray` against `init_image_plane`.
2. **Existing tests**: `caustic_3d_test`, `caustic_ent_test` (a–e), `integrator_compare_test`,
   `integrator_dest_test` (ImagePlane path) all pass unchanged.
3. **Application-level**: `caustic_3d` (81², 2 refinement levels) and `caustic_ent` (121², 3 levels, with and
   without `--entfile`) run before and after; the `CAUSTICS` / `CURVES` / `POINTS` tables and the per-pixel
   images are compared in Python: bitwise equal except `G`/`ENERGY` of refined `caustic_ent` vertices, which may
   differ by the `emit` interpolation error (< 1e-5 relative; reported).
4. Timing of both applications before and after (no regression expected: same operations, no extra copies —
   the class holds its arrays by value, no heap allocation per step).
5. Only then: delete `caustic_bundle_legacy.h` and the regression test's legacy dependency (the test keeps the
   `init_ray` vs `init_image_plane` check and the class self-consistency checks).

## 6  Steps

1. `ImagePlane::init_ray` + `Raytracer::constants_from_p(Ray&)` + `emit_energy`; rerun the imaging/integrator
   tests.
2. Rename the current header to `caustic_bundle_legacy.h` (namespace `legacy`), write the new `caustic_bundle.h`.
3. Regression test (§5.1) → identical.
4. Migrate `caustic_3d.cpp`, then `caustic_ent.cpp`; migrate the two tests; delete `single_ray_bundle.h`.
5. Application-level comparison (§5.3), timing.
6. Remove the legacy header; docs.

## 7  Notes / open points

- `T` stays a template parameter for consistency with `Raytracer<T>`/`ImagePlane<T>`; the applications use
  `double` (the `MinoStepper` constants are only validated in double).
- The bundle keeps a reference to its `ImagePlane` (for `init_ray`, `ray_redshift`, `emit_energy`); the plane
  must outlive the bundles — true in both applications (bundles are stack objects inside the pixel loops).
- `BUNDLE_U_STOP` and the status codes keep their values so existing FITS files read the same.
