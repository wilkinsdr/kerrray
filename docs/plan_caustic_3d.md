# 3D caustic mapping in Kerr spacetime — `caustic_3d`

Design note and results for the `caustic_3d` application (2026-09-10). Supersedes `caustic_imageplane`,
which mapped critical curves on the sky from finite differences of the disc-landing coordinates of
adjacent pixels and only saw the caustic where it met the disc plane.

## 1  Problem

Map the caustic surface of the congruence of null geodesics reaching a distant observer at
Boyer–Lindquist `(r = dist, θ = incl)`: the locus of points where neighbouring rays of the family cross
(Rauch & Blandford 1994, ApJ 421, 46; Bozza 2008, PRD 78, 063014). In Kerr the primary caustic is a tube
with an astroid cross-section displaced from the optical axis; in the Schwarzschild limit it collapses onto
the axis behind the hole, with higher-order caustics alternately in front of and behind it.

## 2  Method

The family is `X(x_img, y_img, τ)` with `τ` the Mino time (any smooth parameter gives the same caustic set,
since adding a multiple of `dX/dτ` to a deviation vector does not change the determinant). The caustic
condition is

    J(τ) = det[ ∂X/∂x_img , ∂X/∂y_img , dX/dτ ] = 0

in the Cartesian coordinates of `cartesian()` (`kerr.h`), which keeps the polar axis and the φ branch cut
harmless.

Per image-plane pixel a **bundle of five rays** — centre `(x, y)` and offsets `(x ± δ, y)`, `(x, y ± δ)` —
is initialised by `ImagePlane` (five planes shifted by `±δ`) and advanced in **lockstep** with the Mino-time
symplectic stepper: identical Mino steps, chosen from the centre ray by the same policy as
`propagate_symplectic` (`mino_step_size()`). At every step `J` is formed from central differences of the
offset rays' positions and the centre ray's Mino-time velocity `(√Δ p_u, p_θ, φ̇)`. A sign change of `J`
across a step is located by **bisection on the fraction of the step** (re-taking the composed step for all
five rays from the saved previous states; 60 iterations, ~1e-13 of a step) and recorded as a caustic
point with the crossing index `N`, the ray's radial turning points and equatorial crossings so far, and
the direction of the sign change.

Only `sign(J)` matters, so the bundle can be kept in the linear regime through photon-shell orbits
(separations grow ~e⁶ per orbit) by **Benettin-style rescaling**: whenever an offset pair's Cartesian
separation from the centre exceeds `delta_max` (default `100 δ`), its deviation in the full state
`(u, p_u, θ, p_θ, t, φ)` *and in the constant of motion `h`* is rescaled to `δ` (φ taken modulo 2π, since
the two members of a pair passing a pole on opposite sides differ by 2π in BL φ). The accumulated log10
scale is stored per caustic point.

Rays continue until the centre ray escapes (`r ≥ r_max`, default `1.1 dist`) or approaches the horizon
(`u < BUNDLE_U_STOP = 0.25`, i.e. `r − r₊ < 0.03 √(1−a²)` — finite differences on the last step before
capture are not trustworthy and produced spurious sign flips). If an offset ray is captured first the bundle
straddles the critical curve and the pixel is flagged `bundle split` (its caustics so far remain valid).

**Parameters** (`par_example/caustic_3d.par_example`): `dist, incl, plane_phi0, spin, x0, xmax, y0, ymax,
Nx, Ny, delta (1e-4), delta_max, r_max, symp_step, symp_order, max_tstep, maxtstep_rlim, max_phistep,
steplim, max_caustics, show_progress`, plus `dump_pixel = ix,iy` / `dump_file` to write the centre ray and
`J` of one bundle at every step (diagnostics). Command-line overrides: `--parfile --outfile --spin --incl
--Nx --Ny --delta --show_progress --dump_pixel --dump_file`.

**Output** (FITS): `CAUSTICS` binary table (`IX IY XIMG YIMG N TAU T R THETA PHI X Y Z RFLIPS EQCROSS DJSIGN
LOGSCALE`) and per-pixel images `NCAUST, STATUS, R1, THETA1, PHI1, EQCROSS, TAU_END` on the
`(Nx+1)×(Ny+1)` grid. Observer frame for plotting: line of sight `n = (sin i cos φ₀, sin i sin φ₀, cos i)`,
image x along `(−sin φ₀, cos φ₀, 0)`, image y along `(−cos i cos φ₀, −cos i sin φ₀, sin i)`.

## 3  Code

- `src/raytracer/mino_stepper.h` — `MinoState<T>`, `MinoStepper<T>` (Verlet substep, Yoshida 4/6
  composition, `hit_horizon`) and `mino_step_size()`, factored out of `Raytracer::propagate_symplectic_impl`
  (behaviour-preserving; all symplectic tests unchanged).
- `src/caustic/caustic_bundle.h` — bundle tracer (`trace_bundle`, `bundle_jacobian`, rescaling, ...),
  shared by the application and the test.
- `src/caustic/caustic_3d.cpp` — application (OpenMP over pixels, FITS output).
- `src/tests/caustic_3d_test.cpp` — Schwarzschild checks, no cfitsio needed.
- `python/caustic_3d.py` (matplotlib: 3D scatter; line-of-sight projections; tube cross-section;
  equatorial-plane intersection; sky maps of crossings and ray fate) and `python/caustic_3d_plotly.py`
  (interactive HTML, one toggleable trace per order).

## 4  ImagePlane fix (affects all image-plane applications)

The caustic tracer exposed an inconsistency in `ImagePlane::init_image_plane`: the constants of motion
`h = −b sin i cos β`, `ℓ_θ = b sin β` (with θ initially *increasing* for `y > 0`) belong to the ray that
passes the black hole on the `(−x, −y)` side, while the ray was placed at `(x, y)` on the plane. The
whole family therefore converged through a focus on the line of sight at `D/2` — the rays were those of a
point observer at distance `D/2` (a pinhole camera: images inverted, hence the `flip_image` option in the
imaging applications, and disc-hit radii off by ~`b/D`, ~1% at `D = 1000`). The Jacobian `J/r²` of the
old family dropped from 1 to ~1e-3 at `r ≈ D/2` and half the rays showed a spurious caustic there.

Fixed (2026-09-10): `h = +b sin i cos β` and θ initially *decreasing* for `y > 0`, so the ray from plane
position `(x, y)` passes the hole on the `(x, y)` side and `J/r²` is constant to 4e-5 down to `r = 50`
(test (c)). Image-plane coordinates now follow Cunningham & Bardeen: `x = α = −h_phys / sin i` (the shadow
of a prograde hole is displaced to `x ≈ +2a sin i`), `y = β` towards the pole. `flip_image` now defaults
to `0` in `imageplane_disc_image*`; existing outputs made with the old code are the images of an observer
at `dist/2`, rotated by 180° (or mirrored in x with `flip_image = 1`).

Also fixed on the way: `ImagePlane` used `dy` for the x spacing (harmless for square pixels), and
`mino_polar_flow()` now advances φ by π for a ray with exactly `h = 0` passing over a pole (previously
such a ray was reflected).

## 5  Verification

`caustic_3d_test` (a = 0, i = 60°, dist = 1000, 41×41 bundles over ±8 rg): all 1334 crossings lie on the
optical axis (max offset 2.0e-3 rg — the residual scales as `1/dist²`: 1.2e-4 at dist = 4000, and is the
finite-distance error of the plane initialisation, not of the tracer); every escaping ray crosses the
primary caustic exactly once, behind the hole, at a distance increasing monotonically with impact
parameter; higher orders alternate in front / behind; primary caustic positions agree to 4e-6 rg between
`δ = 1e-3` and `1e-5` (round-off appears at `δ = 1e-6`; halving `symp_step` and `max_tstep` changes
nothing at the 1e-3 level).

Kerr (a = 0.998, i = 60°, 321×321 bundles over ±8 rg, 69 s on 4 threads, 1.0e5 crossings, up to 10 per
ray): the primary caustic is a thin tube behind the hole displaced to `X' ≈ +1.2 rg` (towards the shadow
side) with a four-cusped (astroid) cross-section ~0.1 rg across at `Z' = −4.75`, curving towards the hole
inside `r ≈ 5`; the secondary caustic (N = 2) forms a fan/tube on the observer's side of the hole extending
to large distance, and the intersection with the equatorial plane is a spiral arc trailing from the ISCO
outwards. Resolving the astroid needs the critical-curve annulus sampled finely (image-plane pixels ≪ 0.05
rg there): zoom the plane (`x0/xmax`) onto the ring rather than raising `Nx` globally.

## 6  Open items

- ~~Adaptive sampling of the image plane near the critical curve~~ — done 2026-09-17 (`refine_levels`,
  `src/caustic/adaptive_plane.h`; see `plan_caustic_ent.md` §6).
- Fold/cusp classification (the astroid cusps) from the second derivative of the map is not attempted; the
  `DJSIGN` column and the per-pixel `NCAUST` give the parity structure.
- A point-observer initialiser (rays converging exactly on `(r, θ)`) would remove the residual `1/dist²`
  offset of the plane initialisation.
