# Caustic / disc-plane intersection in the reverberation energy–time plane — `caustic_ent`

Design note and results for the `caustic_ent` application (planned and implemented 2026-09-17).

## 1  Goal

`caustic_3d` maps the caustic surface of the family of rays reaching a distant observer in 3D. `caustic_ent` traces
the curve(s) where that surface intersects the equatorial disc plane and places each point of the curve in the
energy/time plane of the reverberation response function computed by `disc_ent_line_rbin` (the "ent plot"):

- **energy** `E = E_line / g`, with `g` the total Doppler + gravitational redshift of a photon emitted at that point of
  the disc (Keplerian orbit, `V = -1`) and reaching the observer — evaluated with `Raytracer::ray_redshift` on the
  centre ray of the bundle at its disc crossing, exactly as the imaging codes do (`reverse = true`, `V = -1`);
- **time** `t = t_sd(r, φ) + t_do(r, φ) − t_continuum`, the coordinate time from the point source to that point of
  the disc plus the coordinate time from the disc to the observer's image plane, minus the direct source-to-observer
  time (the continuum arrival), so that the zero point matches the ent code (`t_tot = trf.t + disc_time[ir] −
  t_continuum`; `trf.t` is the raw coordinate time to the image plane at `dist`, and the image-plane rays start at
  `t = 0`, so the centre ray's `t` at the crossing is `t_do` directly). Multiplied by `rgs` when `mass > 0`.

The curves are then over-plotted on the `RESPONSE` image of an ent FITS file (`python/caustic_ent.py`), together
with the same curves on the disc plane and on the image plane so that the features in the ent plot can be
identified with their location on the disc.

## 2  Method

### 2.1  Where the caustic meets the disc

The 3D caustic condition is `J(τ) = det[∂X/∂x, ∂X/∂y, dX/dτ] = 0` along the ray. On the plane `θ = π/2` the disc
landing map `(x, y) → (r, φ)` has derivatives `∂X_disc/∂x = ∂X/∂x + (dX/dτ) ∂τ_land/∂x`, so
`det[∂X_disc/∂x, ∂X_disc/∂y, dX/dτ] = J` and, since the first two vectors lie in the plane, this equals the 2D
Jacobian of the disc map times the normal component of the ray velocity. Hence wherever the ray is not tangent to
the plane, **the caustic meets the disc exactly where `J`, evaluated at the ray's equatorial crossing, vanishes** —
the points of infinite disc magnification.

So, per image-plane pixel, the bundle tracer records `J` at the ray's 1st, 2nd, … `max_eqcross`-th equatorial
crossing (default 3): when `crossed_equator(prev, current)` fires for the centre ray, the composed Mino step is
re-taken from the saved states of all five rays with the fraction found by **bisection on `θ − π/2`** (same
mechanism as the existing caustic bisection, ~1e-13 of a step) and `bundle_jacobian` is evaluated on the landed
bundle. Recorded per crossing `n`: `J_n`, `r, φ, t` of the centre ray, its Mino state `(u, p_u, θ, p_θ)` and
constants `(k, h)` (for the redshift), the caustic index `N` reached so far (so the disc curve can be labelled with
the same order as in `caustic_3d`), and `NaN` if the ray was captured / escaped / split before its `n`-th crossing.
This gives images `J_n(x, y)` on the `(Nx+1)×(Ny+1)` grid — one sheet per crossing order.

### 2.2  Tracing the zero contour

The caustic/disc curve for sheet `n` is the zero contour of `J_n(x, y)`, extracted by **marching squares** on the
pixel grid (edges with a `NaN` end are skipped; saddle cells resolved by the cell-centre sign). Segments are linked
into ordered polylines (`CURVE` id, `VERTEX` index) so they plot as lines rather than scatter.

Each contour vertex lies on a grid edge between pixels of opposite `sign(J_n)`. Linear interpolation of `J_n` gives
its position to ~`dx/|∇J|`; by default the vertex is then **refined by bisection along the edge**: a new bundle is
traced from the interpolated image-plane position up to its `n`-th equatorial crossing only (early stop), `J_n`
evaluated, and the sign change bracketed (`refine_iter`, default 10 → edge resolved to `dx/1024`; `0` disables).
The refined bundle's centre ray provides `r, φ, X, Y, g, t_do` at the caustic point directly — no interpolation of
physical quantities across the pixel, no `|cos θ|` threshold.

Guard against false contours: `J_n` is discontinuous where the number of equatorial crossings of neighbouring rays
changes (rays tangent to the plane, crossings appearing in pairs) and across the boundary of the region where
the `n`-th crossing exists. An edge is accepted only if the two pixels' `n`-th crossing positions differ by less than
`max_jump` (default 1 rg, scaled with the pixel size); rejected edges are counted and reported.

### 2.3  Redshift and times

- `g` at a vertex: `planes[0]->ray_redshift(-1, true, false, r, π/2, φ, k, h, Q, rdot_sign, thetadot_sign, emit)`
  with `Q = carter_Q(θ, p_θ, k, h, a)`, `rdot_sign = sign(p_u)`, `thetadot_sign = sign(p_θ)` from the landed Mino
  state (backward-ray direction, as for `ImagePlane` rays), and `emit` from the pixel's `redshift_start()`. Points
  with `r < r_min` (ISCO by default) get `g = NaN`, `E = NaN` and `INDISC = 0` (the Keplerian velocity is not
  defined there); likewise `r > r_disc`. They are still written so the disc-plane panel shows the full curve.
- `t_do` = centre ray `t` at the crossing (ImagePlane rays start at `t = 0`; `t` increases along the backward ray —
  verified in the test, §5).
- `t_sd(r)`: **either** computed internally — a `PointSource` traced with the same parameters as the ent code
  (`source, V, cosalpha0/max, dcosalpha, beta0/max, dbeta, rmax`, `integrator` for the lamppost rays), rays landing
  on the disc binned in `r` exactly as `disc_ent_line_rbin` does (`Nr, logbin_r, rmin, rdisc`, mean `t` per annulus)
  — **or** read from the `DISC` table (`r`, `time`) of an existing ent FITS given by `entfile` /
  `--entfile=`, in which case `spin, line_en (LINEEN), t_continuum (TSTART), mass/rgs, r_isco` are taken from its
  header and checked against the par file. Between annulus mean times the code interpolates linearly in `r` by
  default (`time_interp = 1`); `0` reproduces the ent code's per-annulus step function. The source is treated as in
  the ent code (azimuthally averaged times per annulus); an off-axis source therefore works "as in the ent plot",
  with the same approximation.
- `t = (t_sd + t_do − t_continuum) · rgs`.

### 2.4  Parameters (`par_example/caustic_ent.par_example`)

Observer / bundle block as `caustic_3d` (`dist, incl, plane_phi0, spin, x0, xmax, y0, ymax, Nx, Ny, delta,
delta_max, r_max, symp_step, symp_order, max_tstep, maxtstep_rlim, max_phistep, steplim, show_progress`), plus
`max_eqcross (3), refine_iter (10), max_jump (1.0)`. Source-time block as `disc_ent_line_rbin` (`source, V,
cosalpha0, cosalphamax, dcosalpha, beta0, betamax, dbeta, rmax, rmin, rdisc, Nr, logbin_r, integrator, rk45_tol,
line_en (6.4), t_continuum, mass (-1), time_interp (1)`), or `entfile`. Command-line overrides: `--parfile --outfile
--entfile --spin --incl --Nx --Ny --delta --t_continuum --show_progress`.

### 2.5  Output (FITS)

- `CURVES` binary table, one row per contour vertex: `SHEET` (equatorial crossing order `n`), `NCAUST` (caustic
  index `N` of the crossing along the ray, as in `caustic_3d`), `CURVE`, `VERTEX`, `XIMG, YIMG`, `R, PHI, X, Y`,
  `G, ENERGY, TDO, TSD, TIME` (`TIME` in the ent convention and units), `INDISC`, `REFINED` (bisection converged),
  `JSLOPE` (sign of `∂J` across the edge, i.e. the fold parity).
- Per-pixel images for each sheet `n = 1..max_eqcross`: `JDISC<n>`, `RDISC<n>`, `PHIDISC<n>`, `GDISC<n>`,
  `TDISC<n>` (disc landing map: `r, φ, g, t_do` at the `n`-th crossing), plus `NEQCROSS` and `STATUS` (as
  `caustic_3d`). These also let the plotting script draw the whole image-plane disc map and the locus of the direct
  image in energy/time as a background check.
- `DISC` table: `r, time` as used for `t_sd` (copied from the ent file or computed).
- Header: all parameters, `TSTART = t_continuum`, `LINEEN`, `RGS`, `ISCO`, `ENTFILE`.

## 3  Code (as built)

- `src/caustic/caustic_bundle.h` — `BundleParams::max_eqcross` / `stop_after_eqcross`, `struct DiscCrossing`,
  and `trace_bundle(..., vector<DiscCrossing>* disc)`: at each equatorial crossing of the centre ray the composed
  step is re-taken by bisection on `θ − π/2` for all five rays and `J`, `t`, `r`, `φ`, the Mino state, `(k, h)`,
  the caustic count so far and the rescaling `logscale` are recorded (the unscaled Jacobian is
  `J·10^logscale`). `disc_crossing_redshift()` evaluates `ray_redshift` at a crossing with the momentum signs
  taken from `(p_u, p_θ)` and `Q = carter_Q`. `caustic_3d` behaviour unchanged (`caustic_3d_test` passes).
- `src/caustic/disc_caustic_contour.h` — `zero_contours()`: NaN-aware marching squares (exact zeros count as
  positive; saddles by the cell-centre average) and linking into ordered open/closed polylines.
- `src/caustic/caustic_ent.cpp` — application: source-time table (internal `PointSource` run with the ent
  code's annulus binning, or `entfile` read with cfitsio), image-plane bundles (OpenMP), contours per sheet,
  per-vertex refinement (OpenMP; each trial position is a fresh `CausticBundle` built from the plane with
  `ImagePlane::init_ray`, since the 2026-09-18 class refactor — see `plan_caustic_bundle_class.md`), FITS output.
- `src/tests/caustic_ent_test.cpp`, `par_example/caustic_ent.par_example`, `python/caustic_ent.py`.
- `src/caustic/CMakeLists.txt` had committed merge-conflict markers, and `src/raytracer/raytracer.cpp` did not
  compile after the merge (the per-run `steplim` parameter of `run_raytrace` shadowed the local default; now
  `effective_steplim = steplim > 0 ? steplim : <integrator default>` in both overloads). Both fixed.

Differences from the plan: the jump test is applied **after** the refinement rather than as an edge
pre-filter — the disc-landing map has very large derivatives on the higher sheets (adjacent pixels at
`dx = 0.17` land tens of rg apart along the outer spiral while the map is perfectly continuous), so an absolute
jump between pixels rejects genuine vertices; the residual jump across the refined bracket
(`dx/2^refine_iter`) separates a genuine zero (jump → 0, even at a fold where the position goes as √s) from a
discontinuity of `J_n`, and is stored as `JUMP` with the flag `ACCEPT` (default `max_jump = 2 rg`). `NCAUST` in
the table is the order `N` of the caustic meeting the disc (`min(ncaust) + 1` over the bracket: the rays on one
side of the vertex cross that caustic before the disc, those on the other side after), matching `N` of
`caustic_3d`.

## 4  Verification

`caustic_ent_test` (no cfitsio; ~5 s): (a) Kerr `a = 0.998, i = 60°`, 273 pixels: the disc-crossing `r, φ, t`
agree with `ImagePlane::run_raytrace(Symplectic, θ_max = π/2)` to 1e-13 and `disc_crossing_redshift` with
`ImagePlane::redshift` to 5e-14; (b) face-on Schwarzschild direct image: `E_disc/E_obs = √(1−2/dist)/√(1−3/r)`
to 1.4e-5 (the `√(1−2/dist)` is the static observer at the finite image-plane distance, common to all
imaging codes); (c) 1414 crossings: `sign(J_disc) = (−1)^ncaust sign(J_plane)` for every one, i.e. `J` at the
disc flips sign exactly at the recorded caustics; (d) marching squares: circle radius to 4e-4 at `dx = 0.1`, one
closed curve; a line reproduced exactly with one open curve through every row; a NaN region leaves a single open
arc. `caustic_3d_test` and `integrator_compare_test` still pass.

Standard geometry (`a = 0.998, i = 60°`, source at `r = 5` on the axis, `±10 rg`, `max_eqcross = 3`,
`refine_iter = 10`): 121² bundles + refinement 13 s, 241² 42 s on 4 threads (Linux VM). Sheet 1 (direct
image) has no zero of `J` except within `r < 1.09` at the horizon, consistent with the `caustic_3d` result that
the primary caustic only reaches the plane there; the `N = 2` caustic meets the disc on sheet 2 along the spiral
arc from the horizon outwards (`r = 1.08 → 66` within the plane), with `g` from ~15 at the inner end down to
1.06 at `r = 27`, and the `N = 3` caustic on sheet 3 inside `r < 2.6`. Against the `|cos θ| < 0.05` points of
`caustic_3d_a0.998_i60_321.fits`: median distance in the plane 0.09 rg for `N = 2` (`d/r` median 0.02, within
the slab thickness), 0.02 rg for `N = 1`. Residual bracket jumps are all < 0.1 rg (no vertex rejected). All
vertices refined (one interpolated at 241²).

In the energy/time plane the `N = 2` curve runs along the low-energy (`g > 1`, receding, strongly redshifted)
edge of the response at early times; `python/caustic_ent.py` draws it over the response of a
`disc_ent_line_rbin` file (checked with a synthetic file of the same layout) or over the density of disc pixels
from the per-sheet maps, with matching curves on the disc plane and the image plane.

## 5  Usage

    cp par_example/caustic_ent.par_example par/caustic_ent.par
    ./bin/caustic_ent --parfile=par/caustic_ent.par --outfile=dat/caustic_ent.fits [--entfile=dat/ent.fits]
    python python/caustic_ent.py dat/caustic_ent.fits [dat/ent.fits] [--rlim 12] [--tmin/--tmax] [--sheets 2]

With `--entfile`, `line_en`, `t_continuum` (`TSTART`), `rgs` and the disc radii are taken from that file and the
source block of the par file is ignored; the file name is recorded in the `ENTFILE` keyword, and
`caustic_ent.py` then reads that file's `RESPONSE` image for the background of the energy/time panel without it
being named again (a file on the command line overrides it; `--no-background` draws the curves alone; the
density of disc pixels is used only when no ent file is known). The build needs `cmake .` re-run (new targets `caustic_ent`,
`caustic_ent_test`).

## 6  Open items / later

- The curves are sampled where the contour crosses pixel edges, so their density follows the image-plane
  resolution (~40 vertices at 121², ~100 at 241²); adaptive refinement of the pixels crossed by the contour
  would give dense curves cheaply.
- Off-axis or moving source: `t_sd(r, φ)` from a 2D binning (the ent code would need the same change).
- Emission inside the ISCO (plunging velocity, `motion` option of `ray_redshift`): currently `G = NaN` there.
- Fold/cusp classification along the disc curve from `JSLOPE` sign changes.
