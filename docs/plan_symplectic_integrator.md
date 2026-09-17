# Plan: symplectic (Mino-time) integrator for `Raytracer`

Status: design + test plan, 2026-09-09. Nothing in the repo has been modified yet.
A Python prototype of the scheme (`mino_symplectic_prototype.py`) accompanies this plan and was used for the numbers quoted below.

## 1. What we are adding

A fourth entry in `enum class Integrator { Euler, RK4, RK45, Symplectic }`, selectable exactly like the others through both `run_raytrace()` overloads (fixed `theta_max` and `RayDestination*`), with the same locals-copy / loop / write-back structure as `propagate_rk45()`, the same status flags, the same `TextOutput` path writing, and the same `Ray<T>` fields on exit so that `redshift()`, the emissivity/imageplane/caustic applications and the Python plotting notebooks need no changes.

The important difference from the three existing methods is *what* is integrated. Euler/RK4/RK45 integrate only the positions and re-derive the four momenta at every stage from the constants of motion `(k, h, Q)` through `momentum_from_consts()`, which is why they need the `rdot_sign`/`thetadot_sign` flip machinery and why their intermediate stages can be pushed inside the horizon. The symplectic method integrates a genuine Hamiltonian system in `(r, p_r, θ, p_θ)`: momenta evolve continuously through turning points, no square roots are taken, no sign tracking is needed, and `Q` becomes a *diagnostic* (a quantity that should be conserved) rather than an input. `k` and `h` remain exact inputs because `t` and `φ` are cyclic.

## 2. Formulation

Conventions follow the codebase: `E = k`, `L = h`, Carter constant `Q`, `a = spin` (negated by `ImagePlane` for backward tracing; everything below is even in the sign of `a` except `P`, which carries it correctly). Mino time is `dλ_M = dλ / ρ²`, with `ρ² = r² + a² cos²θ`.

The Kerr null-geodesic Hamiltonian multiplied by `ρ²` (a legitimate time reparametrisation on the `H = 0` shell) separates completely:

```
H_M = H_r(r, p_r) + H_θ(θ, p_θ) = 0
H_r = ½ Δ p_r²  −  P(r)² / (2Δ)          P = (r² + a²) E − a L,   Δ = r² − 2r + a²
H_θ = ½ p_θ²    +  (L − a E sin²θ)² / (2 sin²θ)
```

`H_r` and `H_θ` are each separately conserved; on the null shell `H_θ = ½[Q + (L − aE)²]` and `H_r = −H_θ`. Standard identities check out against `kerr.h`: `Δ² p_r² = R(r)` and `p_θ² = Θ(θ)`, exactly the `rdotsq`/`thetadotsq` expressions in `momentum_from_consts()` up to the `ρ⁴` factor. The cyclic coordinates advance by

```
dt/dλ_M = (r² + a²) P / Δ + a (L − a E sin²θ)      ( = ρ² · pt   from momentum_from_consts )
dφ/dλ_M = a P / Δ + L / sin²θ − a E                ( = ρ² · pphi from momentum_from_consts )
```

so the existing formulas can be reused verbatim, multiplied by `ρ²` — this is also a free unit test.

### Radial coordinate

The kinetic term `½ Δ(r) p_r²` has a position-dependent mass, which would make a naive leapfrog implicit. The point transformation

```
r = 1 + b cosh u,    b = √(1 − a²),    p_u = √Δ · p_r,    Δ = b² sinh² u
```

is canonical and turns the radial kinetic term into `½ p_u²`, so the drift is trivial (`u += h p_u`), the horizon sits at `u = 0` and can never be crossed by an intermediate stage, and there is no cancellation problem near `r₊` in `float` (which `arccosh((r−1)/b)` would have). `arccosh` is evaluated once per ray at initialisation; each kick needs one `cosh`/`sinh` pair to recover `r` and `√Δ`. For `|a| → 1`, `b → 0` and `u` degenerates (`r = 1 + e^u` limit); guard with `|a| ≤ 0.9999` as the code effectively does already via `kerr_horizon`.

State vector integrated per ray: `(u, p_u, θ, p_θ, t, φ)`.

### Scheme

Kick/drift Störmer–Verlet. The polar potential is split further, `V_θ = h²/(2 sin²θ) + W(θ)`, `W = ½a²k² sin²θ − a k h`, because `½p_θ² + h²/(2 sin²θ)` is the free particle on the unit sphere whose flow is an exact great circle (`cos θ = A sin ψ`, `A = √(1 − h²/J²)`, `J = √(2H_c)`, `ψ = Jλ_M + ψ₀`, with the `h/sin²θ` part of `dφ/dλ_M` integrating to `sign(h)·atan(|h|/J · tan ψ)`). Using that exact flow as the polar drift is what makes near-axis sources work: the unit test showed the naive `θ += h p_θ` split *diverging* for a lamppost at `θ = 10⁻³` even at `h = 5×10⁻⁴`, while the great-circle split is accurate to `10⁻⁸` at `h₀ = 0.01`. Item 4 of §6 is therefore resolved from the start rather than deferred.

```
kick(h):  p_u −= h · dV_r/dr · b sinh u ;  p_θ −= h · dW/dθ ;
          t   += h · dt/dλ_M ;             φ   += h · (dφ/dλ_M − h_z/sin²θ)
drift(h): u   += h · p_u ;                 (θ, p_θ, φ) ← great-circle flow for Mino time h   [mino_polar_flow()]
verlet(h) = kick(h/2) · drift(h) · kick(h/2)
```

Helpers implemented in `kerr.h` (2026-09-09): `mino_b`, `mino_u_from_r`, `mino_r_from_u`, `mino_tdot`, `mino_phidot`, `mino_dVr_dr`, `mino_dVtheta_dtheta` (diagnostic), `mino_dW_dtheta`, `mino_polar_flow`, `mino_init`, `mino_to_bl`, `mino_hamiltonian_r/theta`, `carter_Q`. The steppers in `src/tests/symplectic_unit_test.cpp` (`mino_kick`, `mino_drift`, `mino_verlet`, `mino_yoshida4`) are the reference implementation the propagator should copy.

`t` and `φ` are updated inside the kick because they are the `∂H/∂p_t`, `∂H/∂p_φ` flows of `V`, evaluated at frozen `(r, θ)`; this keeps the composition exactly symplectic in the extended phase space, so `t` and `φ` inherit the full order of the scheme. Higher order by composition: Yoshida 4th order (3 Verlet substeps, coefficients `w₁ = 1/(2−2^{1/3})`, `w₀ = −2^{1/3}/(2−2^{1/3})`) as the default, 2nd (plain Verlet) and 6th (7 substeps, Yoshida 1990 solution A) selectable. Cost per step: order 4 = 3 kicks + 3 drifts ≈ 3 "function evaluations", each cheaper than one `momentum_from_consts()` call (no `sqrt`, but one `cosh/sinh`, one `sin/cos`).

### Step size

Symplecticity wants a fixed step, and a fixed Mino step is naturally adaptive near the hole (affine step `= ρ² h`). But Mino time reaches `r = ∞` in *finite* λ_M (`dr/dλ_M ≈ E r²`), so a fixed Mino step is catastrophic far out — the prototype blew up at `r ≈ 1000` with any fixed `h`. Far from the hole the two split pieces are individually `O(E² r²)` and cancel to give `H_M = 0`, so the splitting error there scales as (affine step)² for Verlet and (affine step)⁴ for Yoshida-4. The remedy that also keeps the user-facing knobs unchanged is to reuse the existing coordinate-time cap:

```
h = min( h₀ ,  max_tstep / |dt/dλ_M| ,  max_phistep / |dφ/dλ_M| )
```

`max_tstep / |dt/dλ_M| ≈ max_tstep / (E r²)` at large `r`, i.e. a constant affine step of `max_tstep` (the same meaning `MAXDT` already has). For this integrator the `MAXDT_RLIM = 100` cut-off must *not* be applied — the cap has to hold all the way to `r_max`. Near the hole `h = h₀` is constant and the integration is exactly symplectic there, which is where multi-orbit photon-sphere rays live; the step only varies in the smooth far-field escape/approach phase. Suggested mapping `h₀ = 1/precision` (so `precision = 100` → `h₀ = 0.01`), overridable via `set_symplectic_step(h0)`.

Cost consequence to be measured: the far field costs `≈ r_max / max_tstep` steps (≈ 1000–4000 for `r_max = 1000`), versus a few hundred for Euler/RK4 whose far-field step grows with `r`. Options if that turns out to dominate run time: (a) a larger `max_tstep` for `r > 100` accepted as an `O((max_tstep)⁴)` error in `H_r`; (b) hand over to RK45 outside `r_sym` (hybrid; more code); (c) a better far-field splitting (future work — the `E`-part of `V_r` cannot be absorbed into an exactly solvable flow, it is elliptic).

### Termination and boundaries

Loop condition and status flags identical to `propagate_rk45()`. Horizon: stop when `r ≤ horizon` (`set_boundary` still works because the check is on `r`, not `u`); an infalling ray has `p_u → −∞` as `u → 0`, so the last step overshoots to `u < 0` — clamp `u = 0` and flag `RAY_STATUS_HORIZON`. `RLIM`: `r ≥ rlim` as now (optionally clamp the last step, see §6). `thetalim` / `RayDestination`: because `θ` moves linearly inside a drift, the sub-step fraction at which `θ` reaches `θ_lim` is exact (`(θ_lim − θ)/(h p_θ)`), so the disc landing can be made exact rather than the linear-extrapolation clamp the RK methods use. Simplest first implementation: mirror the RK45 pattern — convert `dest->step_limit(r, θ, φ, ṙ, θ̇, φ̇)` (which takes and returns affine quantities) by passing `ṙ = Δ p_r/ρ² = b sinh u · p_u/ρ²`, `θ̇ = p_θ/ρ²` and dividing the returned affine step by `ρ²`; clamp the last step and accept the small overshoot. Second implementation: exact sub-step landing (a clamped or shortened final step breaks symplecticity for one step only, which is irrelevant).

`ERGO` and `NEG_ENERGY` checks reuse the existing expressions with `pt = (dt/dλ_M)/ρ²`, `pphi = (dφ/dλ_M)/ρ²`. `equatorial_crossings` counted exactly as now. `rdot_flips` = number of sign changes of `p_u`. On exit write back `rays[ray].pr = b sinh u · p_u/ρ²`, `ptheta = p_θ/ρ²`, `pt`, `pphi` as above, `rdot_sign = sign(p_u)`, `thetadot_sign = sign(p_θ)` — these are what `ray_redshift()` consumes, so redshifts are unchanged in meaning. `MIN_STEP` is not needed; a new `SYMP_STEPLIM` (start at `1,000,000`, tune from the benchmark) replaces `RK45_STEPLIM` in `run_raytrace()`.

## 3. Prototype results (Python, double precision)

Near-critical image-plane ray for `a = 0.998` (critical curve at photon-orbit radius `r_ph = 2`, `L` offset by `ε`), started at `r = 1000, θ = 1.2`, reference = DOP853 at `rtol 1e-13` on the same Mino-time ODEs. `dθ, dφ` are exit-direction errors after 3.5 orbits (`ε = 10⁻³`); `max|dH_r|`, `max|dH_θ|` are the worst violations of the two separately conserved pieces along the ray (`H_M = 0` is their sum).

| order | h₀ | max_tstep | steps | dθ | dφ | max\|dH_r\| | max\|dH_θ\| |
|---|---|---|---|---|---|---|---|
| 2 | 0.01 | 1.0 | 2089 | 2.2 | 18 | 3.6 | 2e-2 |
| 2 | 0.001 | 0.1 | 22516 | 2.6 | 6.7 | 4e-2 | 2e-4 |
| 4 | 0.01 | 1.0 | 2330 | 5e-2 | 0.7 | 5e-4 | 6e-4 |
| 4 | 0.01 | 0.25 | 8385 | 1e-3 | 6e-3 | 4e-6 | 6e-4 |
| 4 | 0.0025 | 0.25 | 9335 | 5e-4 | 3e-3 | 2e-6 | 3e-6 |
| 4 | 0.001 | 0.1 | 23337 | 6e-6 | 4e-5 | 1e-7 | 7e-8 |
| 6 | 0.01 | 0.25 | 8385 | 1e-7 | 1e-5 | 5e-8 | 4e-6 |

Take-aways: plain Verlet is not competitive for this problem (the far-field cancellation gives `O(1)` errors in `H_r` unless the affine step is tiny); Yoshida-4 with `h₀ ≈ 0.0025–0.01`, `max_tstep ≈ 0.25` is the working point to benchmark first; the `H_r` error is set by `max_tstep` (∝ step⁴), the `H_θ` error by `h₀` (∝ h₀⁴), so the two knobs are independent and can be tuned separately; order 6 is very cheap extra accuracy per step and worth exposing. With `ε = 10⁻⁵` (5.4 orbits) all methods including the reference are dominated by chaotic amplification, which is the regime the photon-ring stress test in §5 targets. For reference the DOP853 solver needed 6.8k function evaluations at `1e-13`, so raw efficiency per evaluation is *not* where this method wins; robustness (no sign flips, no rejected steps, no horizon-crossing stages, bounded long-term errors) is.

## 4. Code changes

`src/raytracer/raytracer.h`
- `enum class Integrator { Euler, RK4, RK45, Symplectic };`
- `#define SYMP_STEPLIM 1000000`
- members `symp_step` (h₀, default `1/precision`, or `0` meaning derive from `precision`), `symp_order` (2/4/6, default 4); setters/getters `set_symplectic_step()`, `set_symplectic_order()`, following `set_rk45_tol()`.
- two `propagate_symplectic()` declarations with the same argument lists as the two `propagate_rk45()` overloads.

`src/raytracer/raytracer.cpp`
- both `run_raytrace()` overloads: fourth `names[]` entry, `steplim` selection, fourth `case` in both the serial and OpenMP `switch`es.
- `propagate_symplectic()` ×2, laid out like `propagate_rk45()`: copy locals from `rays[ray]`; initialise `(u, p_u, p_θ)` from `(r, θ, k, h, Q, rdot_sign, thetadot_sign)`; main loop = step-size choice, one composed step, boundary/status checks, optional `TextOutput` write; write-back block identical in shape to the RK45 one.
- keep the composition coefficients as `static constexpr` tables next to the DOPRI5 tableau for symmetry.

`src/include/kerr.h` — small inline helpers so the two overloads share no duplicated physics:
- `mino_potential_derivs(u, θ, k, h, a, b, &dVr_du, &dVth_dth, &r, &sqrtDelta)`
- `mino_tdot_phidot(r, θ, k, h, a, &tdot, &phidot)`
- `mino_from_bl(...)` / `bl_from_mino(...)` for the covariant↔contravariant conversions at entry and exit
- `mino_hamiltonian_r/θ(...)` and `carter_Q(...)` for diagnostics (used by the tests).

Applications: add `"symplectic"` to the `integrator` string parsing in `imageplane_disc_image.cpp`, `imageplane_disc_image_isco.cpp`, `caustic_imageplane.cpp` (and read optional `symp_step`, `symp_order` parameters next to the existing `rk45_tol` handling); update the corresponding `par_example/` files. `emissivity.cpp` is hard-wired to RK45 — leave it, or give it the same `integrator` parameter while we are there.

Housekeeping: `CLAUDE.md` integrator table and description; `docs/session_<date>.md` in the existing style; `src/tests/CMakeLists.txt` entries for the new tests.

Not changed: `Ray<T>` struct (diagnostics are recomputed from the final state in the tests instead of stored per ray), `RayDestination` interface (the affine-step contract of `step_limit()` is honoured by converting), `redshift*()`.

## 5. Test plan

All test executables go in `src/tests/`, outputs in `dat/`, plots via Python scripts beside the existing `emissivity_rk45_*.py`.

**T1 — unit tests (`symplectic_unit_test`).** (a) Initial-condition round trip: from a `Ray` set up by `PointSource`/`ImagePlane`, build `(u, p_u, p_θ)`, convert back to `(ṙ, θ̇, ṫ, φ̇)` and compare with `momentum_from_consts()` to `1e-12`. (b) `dt/dλ_M = ρ²·pt`, `dφ/dλ_M = ρ²·pphi` at random `(r, θ)`. (c) `H_M = 0` at initialisation for every ray in a grid. (d) Convergence order: integrate one ray to a fixed λ_M with `h, h/2, h/4`; error ratios ≈ 4 (order 2), 16 (order 4), 64 (order 6). (e) Time reversibility: integrate forward N steps then backward N steps, recover the initial state to round-off. (f) Same for `Raytracer<float>`, with looser tolerances.

**T2 — cross-integrator consistency (`integrator_compare_test`, generalising `raytrace_rk4_test`).** Same lamppost grid through all four methods plus an RK45 reference at `rk45_tol = 1e-12`. Check termination class agreement (disc / rlim / horizon) and report percentiles of the differences at the disc in `r`, `φ`, `t` and in `redshift()` (the `ray_redshift` path exercises `rdot_sign`/`thetadot_sign` write-back). Pass criteria applied only to non-separatrix rays, following the `MIN_BIN_RAYS` logic already in `emissivity_rk45_test`.

**T3 — work–precision benchmark (`integrator_perf_test` extended + `integrator_sweep.py`).** For each method sweep its accuracy knob — Euler/RK4 `precision ∈ {50,100,200,400,800}`, RK45 `tol ∈ {1e-6 … 1e-12}`, Symplectic `h₀ ∈ {0.02 … 0.001} × max_tstep ∈ {1, 0.5, 0.25, 0.1} × order ∈ {2,4,6}` — on two ray sets: the lamppost grid (`r_s = 5`, `a = 0.998`) and an `ImagePlane` grid at 60° inclination. Record wall time (OpenMP thread count pinned, propagation phase only), steps and function-evaluation counts, and errors against the reference: disc landing `(r, φ)`, arrival time, redshift, and for the symplectic runs `max|ΔH_r|, max|ΔH_θ|, max|ΔQ|`. Output CSV to `dat/`, plot error-vs-time Pareto curves. This is the plot that answers "is it faster at equal accuracy".

**T4 — photon-ring stress (`photon_ring_test`).** Rays with impact parameters offset from the analytic critical curve by `ε = 10⁻², 10⁻³, …, 10⁻⁸` (the `λ_c(r_ph), η_c(r_ph)` formulas in the prototype), for several `r_ph` and spins. Measure winding number and exit direction against the reference, growth of `|ΔH_r|`, `|ΔH_θ|`, `|ΔQ|` with λ_M, the fraction of rays hitting `STEPLIM`, and (for RK45) the number of rejected steps. This is where bounded long-term error should show up, and it is the regime that matters for the caustic/winding-number work.

**T5 — application regression.** `emissivity_rk45_test` extended to RK45-vs-Symplectic on per-annulus emissivity, mean redshift, mean arrival time. `imageplane_disc_image` and `imageplane_disc_image_isco` run with `integrator = rk45` and `= symplectic` on the same par file; a short Python script diffs the FITS redshift/flux images (relative difference maps, histogram of pixel differences) and checks that the same pixels are flagged as horizon/rlim. `caustic_imageplane` compared on the Jacobian and winding-number maps, which are the most step-size-sensitive outputs.

**T6 — speed.** Per-ray microseconds and total wall time for each method at *matched accuracy* (taken from the T3 Pareto curves) on the two grids, single-thread and full OpenMP; plus a profile of one symplectic ray to confirm the far-field steps are the dominant cost and decide whether the §2 far-field options are needed.

Existing tests (`raytrace_rk4_test`, `emissivity_rk45_test`, `integrator_perf_test`) must still build and pass unchanged for the other three methods.

## 6. Known limitations and decisions to make

1. Position-dependent step in the far field means the scheme is exactly symplectic only for `r ≲ (max_tstep/h₀)^{1/2}` (≈ 5 for `h₀ = 0.01, max_tstep = 0.25`). Acceptable by design — that region contains the photon sphere and the disc — but worth stating in the docs.
2. Far-field cost (~`r_max/max_tstep` steps). Measure before optimising.
3. `r_lim` overshoot: the RK45 `rlim` clamp is commented out; with `max_tstep = 1` the symplectic method overshoots `r_max` by up to one step, which shows up as an `O(max_tstep)` error in `t` at the outer boundary. Irrelevant for disc quantities but should be clamped for `trace_rays` output.
4. Near-axis sources with `L ≠ 0` — **resolved** by the exact great-circle polar flow (§2, `mino_polar_flow()`), verified by unit test (g2).
5. Naming — **decided (Dan, 2026-09-09):** `Integrator::Symplectic`, with the order as a parameter (`set_symplectic_order(2|4|6)`, par-file `symp_order`).
6. `emissivity.cpp` — **decided:** gets the `integrator` parameter now (default `rk45`, so existing par files behave as before), plus `symp_step`/`symp_order`.
7. Disc landing — **decided:** use the exact in-drift sub-step from the start. Inside the drift that would cross `θ_lim`, take the fraction `f = (θ_lim − θ)/(h p_θ)` of that drift, then finish the composed step's remaining kick with `f`-scaled weight so `t` and `φ` land consistently, and terminate. For a `RayDestination`, apply the same sub-step whenever `dest->reached()` becomes true across a drift and the destination provides a `step_limit()`; fall back to the RK-style clamp for destinations that do not.

## 7. Suggested order of work

1. `kerr.h` helpers + T1 unit test (validates the physics mapping before touching the propagator). **Done 2026-09-09** — `symplectic_unit_test` passes all checks (a)–(g); built in the Linux VM with `g++ -O2 -std=c++17 -fopenmp -I src src/tests/symplectic_unit_test.cpp src/raytracer/raytracer.cpp src/raytracer/pointsource.cpp`, CMake target added.
2. `propagate_symplectic()` for the `thetalim` overload, order 4 only, `run_raytrace` wiring; T2. **Done 2026-09-09** (orders 2/4/6 all in; `integrator_compare_test` passes). See §8 for what the comparison turned up.
3. `RayDestination` overload; application `integrator` parsing; T5 on imageplane. **Done 2026-09-09** except the FITS-level regression, which needs the Mac build (no cfitsio in the Linux VM): `python/imageplane_integrator_diff.py` is ready — run the same par file with `integrator = rk45` and `integrator = symplectic` (avoid a grid point at exactly `x = y = 0`, see §8) and pass the two FITS files to it. Both overloads share `propagate_symplectic_impl()`; `integrator_dest_test` (ImagePlane, 60°, `DiscWithISCODestination`) verifies exact landing (`|θ − π/2| ≤ 2×10⁻¹⁵`), identical results from the two overloads for rays landing outside the ISCO, and 99.96% termination agreement with RK45 at `10⁻¹⁰` (median `|dr| = 3×10⁻⁵`, median `|dg| = 10⁻⁶`). `emissivity` now takes `integrator`/`rk45_tol`/`symp_*`/`max_tstep` (default `rk45`); a pre-existing compile error in its linear-bin area calculation (`disc_r + dr`) was fixed on the way. Cost note: image-plane rays start at `dist`, so the far field dominates — at `dist = 1000`, `max_tstep = 0.25` the symplectic run takes ~4000 steps/ray and is ~2× *slower* than RK45; the par-file examples use `dist = 10000`, where it would be ~40 000 steps/ray. The far-field step policy (§2 options; also letting `max_tstep` grow ∝ r beyond `MAXDT_RLIM`, which is the existing Euler semantics) is the first thing T3 must settle.
4. Orders 2 and 6, setters, par parameters; T3 and T6 benchmarks; decide on far-field handling. **Done 2026-09-09** — see §9.
5. T4 photon-ring stress; docs, `CLAUDE.md`, session notes. **Done 2026-09-09** — see §10. Remaining: FITS-level regression on the Mac build (`python/imageplane_integrator_diff.py`), and a commit.

## 8. Findings from step 2 (cross-integrator comparison, lamppost r = 5, θ = 10⁻³, a = 0.998, 5040 rays)

Implementation notes that differ from §2 as first written:
- Horizon: the step is stopped and the ray flagged `RAY_STATUS_HORIZON` as soon as any drift takes `u ≤ SYMP_U_HORIZON = 0.05` (`r − r₊ ≈ 10⁻³ b`). The potential is singular at `u = 0` and the backward substeps of the Yoshida compositions can otherwise throw a ray that has just reached the horizon back out to `r_max` (seen for 10/252 plunging rays with order 4 before the guard). Photon turning points never lie below `u ≈ 0.5`, so this is unambiguous for any spin.
- The `max_tstep`/`max_phistep` caps are applied only for `r > 2 r₊`. Near the horizon `dt/dλ_M` diverges, and the cap would make an infalling ray creep towards the horizon in ever smaller steps without reaching it.
- Exact boundary landing is done by bisection on the length of the *full composed step* (predicate: θ-limit crossed, or `r ≥ rlim`, or horizon), then snapping the crossed coordinate. This is simpler than sub-step landing inside the drift, works for negative substeps, and carries over unchanged to `RayDestination::reached()`. It fixes the `r_lim` overshoot (§6 item 3).
- Non-finite state → last good state restored, ray flagged `RAY_STATUS_STEPLIM` (failed). Not observed in the comparison.

Results (`integrator_compare_test`, reference = RK45 at `tol = 10⁻¹²`, statistics over rays reaching the disc in both):

| method | time | steps/ray | median \|dr\| | 90% \|dr\| | median \|dg\| | 90% \|dg\| | diverged | max \|H_M\| |
|---|---|---|---|---|---|---|---|---|
| Euler (prec 100) | 0.02 s | 488 | 7e-3 | 3e-2 | 1.5e-3 | 2.8e-3 | 1340/2783 | — |
| RK4 (prec 100) | 0.07 s | 494 | 2e-4 | 2e-2 | 1.3e-5 | 9e-4 | 250/2771 | — |
| RK45 (tol 1e-8) | 1.38 s | 5392 | 3e-5 | 4e-3 | 6e-6 | 1.6e-4 | 4/2865 | — |
| Symplectic o4 (h₀ 0.01, max_tstep 0.25) | 0.56 s | 1286 | 7e-5 | 1.9e-3 | 8e-6 | 3.7e-5 | 0/2890 | 1e-5 |
| Symplectic o6 (same) | 1.20 s | 1281 | 7e-5 | 1.9e-3 | 8e-6 | 3.7e-5 | 0/2890 | 4e-7 |

(4 threads; times are propagation only.) At these settings the symplectic integrator is 2.5× faster than RK45 with better redshift accuracy at the 90th percentile and no diverged rays; a proper work–precision sweep is T3.

**Defect found in the existing Carter-equation propagators.** 220 rays (4.4% of the grid) land at radii ~70% off in Euler, RK4 *and* RK45 at any tolerance, while the symplectic result agrees with a direct Carter quadrature and with an independent scipy integration to 10⁻⁴ (e.g. `cos α = −0.1, β = 0.024`: existing code `r = 14.26`, correct `r = 8.43`). They are exactly the rays with `|cos α| ≤ 0.1` (emitted near a radial turning point) that initially head towards the polar axis (`sin β > 0`); all 220 satisfy both conditions. Likely mechanism: when the polar sign flip fires, the propagators use `|Θ|` in place of `Θ` inside the polar forbidden zone, which corrupts `rdotsq = (k ṫ − h φ̇ − ρ² θ̇²)Δ/ρ²` enough to trigger a spurious radial sign flip when the true `R(r)` is small. Same flip count (1) in both, but the radial turning point is taken ~0.04 Mino-time too early. This affects the standard lamppost geometry (`θ_source = 10⁻³`) and therefore emissivity profiles at the few-per-cent-of-rays level; the test excludes this class from its pass criteria and reports it separately. Two further regimes where the reference itself is unreliable and the symplectic runs (orders 4 and 6, `h₀ = 0.01` vs `0.002`) agree with each other to 10⁻⁶ while RK45 does not: `φ` for rays passing within ~10⁻³ of the pole (RK45 returned `φ = 7630` for one ray with `h = −8×10⁻⁶`), and `φ` at disc crossings inside the ISCO near the horizon, where frame dragging amplifies the reference's own `r` error. RK45 also stalls at its step limit on all 252 plunging rays of this grid (Euler and the symplectic method send them through the horizon).

Two more pre-existing issues seen in step 3: `ImagePlane` produces NaN constants of motion for the ray at exactly `x = y = 0`, and the RK45 adaptive loop never terminates on a NaN state (`max(NaN, MIN_STEP)` is NaN, so the shrink-and-retry loop spins forever) — the symplectic integrator flags that ray as failed (`steps = −1`) and carries on; grids with an even number of intervals centred on the origin hit this. And `emissivity.cpp` did not compile at HEAD (`disc_r + dr`, pointer plus double, in the linear-bin branch of the area calculation) — fixed to `disc_r[ir] + dr`.

## 9. Work–precision sweep (T3/T6) and chosen defaults

`src/tests/integrator_sweep.cpp` runs Euler/RK4 (precision 50–1600), RK45 (tol 10⁻⁶–10⁻¹¹) and the symplectic integrator (orders 4, 6 × `h₀` 0.02–0.0025 × `max_tstep` 1, 0.25, 0.1 × far-field policy) on the lamppost set (5040 rays, defect class excluded) and an image-plane set (`dist = 1000`, 60°, 2500 rays), against a symplectic order-6 reference at `h₀ = 0.001`, `max_tstep = 0.05`. The reference was validated independently: four rays of different types (with and without radial turning points, near-pole passage) agree with direct Carter quadratures using the full-precision constants of motion to `|dr| ≤ 2×10⁻⁷`, `|dt| ≤ 6×10⁻⁷`. Results in `dat/integrator_sweep_{lamppost,imageplane}.{csv,png}`; `python/integrator_sweep_plot.py` draws the Pareto plots (4 threads; times are propagation only).

**Far-field policy (decided).** The `max_tstep` cap now grows ∝ `r` beyond `maxtstep_rlim` (default `MAXDT_RLIM = 100`) — the same semantics as the Euler/RK4 step heuristic — so the far field costs `O(log r_max)` steps instead of `r_max/max_tstep`. On the lamppost set this is free (disc rays never reach `r > 100`); on the image plane it costs ~3× in error for 3× fewer steps, which the order-6 composition more than recovers. Pass a huge `rlim` to `set_max_tstep()` to keep the constant cap.

**The Carter-equation methods have accuracy floors.** Euler, RK4 and RK45 barely respond to their accuracy knobs: on the lamppost set RK4 has median `|dr| = 2.1×10⁻⁴` and 9% diverged rays at *every* precision from 50 to 1600, and RK45 takes ~5400 steps/ray with median `|dr| ≈ 6×10⁻⁵` at every tolerance from 10⁻⁶ to 10⁻¹¹. The step is dominated by `MIN_STEP = 10⁻³`, `MAXDT`, `MAXDPHI` and the linear disc-landing clamp rather than by precision/tolerance. The symplectic runs converge cleanly (`∝ h₀⁴`/`h₀⁶` in the strong field, `∝ max_tstep⁴`/`⁶` in the far field) down to the reference floor of a few 10⁻⁹.

| set | method / setting | steps/ray | time | median \|dr\| | 90% \|dg\| |
|---|---|---|---|---|---|
| lamppost | RK4, precision 1600 | 3304 | 0.54 s | 2.1e-4 | 9.3e-4 |
| lamppost | RK45, tol 1e-10 (best) | 5385 | 1.40 s | 6.0e-5 | 2.4e-5 |
| lamppost | Symplectic o4, h₀ 0.01, max_tstep 0.25, grow | 475 | 0.23 s | 2.6e-7 | 5.0e-8 |
| lamppost | **Symplectic o6, h₀ 0.01, max_tstep 1, grow (default)** | 150 | 0.21 s | 1.2e-10 | 1.2e-10 |
| image plane | RK4, precision 1600 | 3549 | 0.27 s | 7.1e-5 | 2.1e-5 |
| image plane | RK45, tol 1e-11 (best) | 2758 | 0.37 s | 1.2e-5 | 1.6e-5 |
| image plane | Symplectic o4, h₀ 0.01, max_tstep 0.25, grow | 1359 | 0.29 s | 3.8e-7 | 1.8e-7 |
| image plane | **Symplectic o6, h₀ 0.01, max_tstep 1, grow (default)** | 347 | 0.20 s | 3.4e-8 | 1.7e-8 |

**Defaults set accordingly:** `symp_order = 6`, `max_tstep = 1` with growth beyond `r = 100`, `h₀ = 1/precision`. At those settings the symplectic integrator is 7× faster than RK45 on the lamppost set and 1.8× faster on the image plane while being 10³–10⁵× more accurate at the disc, with no diverged rays. For photon-ring / caustic work `max_tstep = 0.25` and `h₀ = 0.0025` are the knobs to tighten.

## 10. Photon-ring stress test (T4)

`src/tests/photon_ring_test.cpp` sets up 140 rays directly from constants of motion offset by `ε = ±10⁻²…10⁻⁸` from the Bardeen critical curve (`λ_c(r_ph)`, `η_c(r_ph)`) at `r_ph ∈ {1.5, 2, 2.5, 3, 3.5}` for `a = 0.998`, started at `r = 1000`, 60°, both polar directions, and traces them with no polar stopping condition until escape (`r = 1500`) or capture. The reference has up to 13 equatorial crossings.

**The chaos floor.** Photon-sphere orbits are unstable with `γ T_orbit ≈ 6` (Schwarzschild), i.e. errors grow ~500× per orbit, so beyond 2–3 orbits no double-precision integrator reproduces another's fate, winding number or exit direction ray by ray. The test makes this visible by comparing the reference with a second reference at a slightly different step: they agree on fate 18–19/20 and on crossings 9–18/20 for `ε ≤ 10⁻⁴`. Agreement at small `ε` has to be read against that row; the pass criteria therefore require 100% fate and winding-number agreement only for `ε ≥ 10⁻³` (≲ 1–2 orbits).

| method | steps/ray | time (140 rays) | stalled | fate ok, ε=10⁻³ | crossings ok, ε=10⁻², 10⁻³ | median exit-angle err, ε=10⁻² |
|---|---|---|---|---|---|---|
| reference self-consistency | 56 617 | 1.5 s | 0 | 20/20 | 20/20, 20/20 | 3e-6 |
| RK45 tol 1e-8 | 18 497 | 0.14 s | 24 | 18/20 | 16/20, 14/20 | 1.8e-2 |
| RK45 tol 1e-10 | 21 293 | 0.16 s | 26 | 18/20 | 20/20, 14/20 | 5.9e-4 |
| **Symplectic default (o6, h₀ 0.01, max_tstep 1)** | 1 047 | 0.03 s | 0 | 20/20 | 20/20, 20/20 | 1.6e-4 |
| Symplectic tight (o6, h₀ 0.0025, max_tstep 0.25) | 3 942 | 0.11 s | 0 | 20/20 | 20/20, 20/20 | 8e-6 |

For `ε ≤ 10⁻⁴` both symplectic runs sit at the chaos floor (e.g. fate 17–19/20, crossings 15–18/20 at `10⁻⁵`–`10⁻⁶`, indistinguishable from the reference's self-consistency), with `|H_M| ≤ 1.3×10⁻⁷` and `|ΔQ|/Q ≤ 3×10⁻¹²` after 13 crossings and no stalled rays. RK45 gets winding numbers wrong for 4/20 rays after a single orbit (`ε = 10⁻²`) and 6/20 at `10⁻³`, and stalls on a sixth of the rays — for winding-number and caustic work the existing propagator is not reliable beyond one orbit, while the symplectic default is exact to the chaos limit at 20× less cost.

Fixed on the way: after an exact landing on `rlim`, `u → r` round-off could leave `r` marginally below `rlim` and the ray re-crossed the boundary every step until the step limit; the propagator now sets `r = rlim` and stops.

Side note: `Raytracer::set_precision()` assigns its parameter to itself (`precision = precision;`) — a no-op; pre-existing, untouched.
