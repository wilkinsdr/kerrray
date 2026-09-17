"""
Compare two FITS outputs of imageplane_disc_image / imageplane_disc_image_isco /
caustic_imageplane produced with different integrators (e.g. rk45 vs symplectic) on
the same parameter file.

Usage:
  python python/imageplane_integrator_diff.py dat/run_rk45.fits dat/run_symplectic.fits [dat/diff.png]

For every image extension present in both files it prints
  - the fraction of pixels that are "hit" (non-zero) in one file but not the other
  - median / 90th-percentile / max of the absolute difference over pixels hit in both
  - the same for the relative difference where the reference value is non-zero
and for angle-like extensions (PHI) the difference is wrapped to [-pi, pi].  A figure with
the difference maps of the main extensions is written to the optional third argument
(default dat/imageplane_integrator_diff.png).

Exit status is 0; this script is a diagnostic, not a pass/fail test -- separatrix rays
(photon-ring pixels) legitimately differ between integrators, so judge the maps, the
percentiles, and the hit-mismatch fraction rather than the maximum.
"""
import sys
import numpy as np
import astropy.io.fits as pyfits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

file_a, file_b = sys.argv[1], sys.argv[2]
out_png = sys.argv[3] if len(sys.argv) > 3 else "dat/imageplane_integrator_diff.png"

ANGLE_EXTS = {"PHI"}
PLOT_EXTS = ["FLUX", "RADIUS", "ENSHIFT", "REDSHIFT", "TIME", "PHI", "ORDER"]


def load(path):
    out = {}
    with pyfits.open(path) as f:
        for hdu in f[1:]:
            name = hdu.header.get("EXTNAME", "").strip()
            if name and hdu.data is not None:
                out[name] = np.array(hdu.data, dtype=float)
    return out


a = load(file_a)
b = load(file_b)
common = [k for k in a if k in b]
print(f"{file_a}  vs  {file_b}")
print(f"common image extensions: {', '.join(common)}")
print()
print(f"{'ext':10s} {'hit A only':>10s} {'hit B only':>10s} {'both':>8s} | {'|diff| med':>10s} {'90%':>10s} {'max':>10s} | {'rel med':>10s} {'rel 90%':>10s}")

stats = {}
for k in common:
    A, B = a[k], b[k]
    if A.shape != B.shape:
        print(f"{k:10s} shape mismatch {A.shape} vs {B.shape}")
        continue
    hitA, hitB = A != 0, B != 0
    both = hitA & hitB
    d = B - A
    if k in ANGLE_EXTS:
        d = np.arctan2(np.sin(d), np.cos(d))
    ad = np.abs(d[both])
    rel = ad / np.abs(A[both]) if both.any() else np.array([])
    nz = np.abs(A[both]) > 0
    rel = rel[nz] if rel.size else rel
    pct = lambda v, p: (np.percentile(v, p) if v.size else float("nan"))
    print(f"{k:10s} {int((hitA & ~hitB).sum()):10d} {int((hitB & ~hitA).sum()):10d} {int(both.sum()):8d} | "
          f"{pct(ad, 50):10.3e} {pct(ad, 90):10.3e} {(ad.max() if ad.size else float('nan')):10.3e} | "
          f"{pct(rel, 50):10.3e} {pct(rel, 90):10.3e}")
    stats[k] = (d, both)

plot_keys = [k for k in PLOT_EXTS if k in stats]
if plot_keys:
    n = len(plot_keys)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for i, k in enumerate(plot_keys):
        d, both = stats[k]
        ref = a[k]
        ax = axes[0, i]
        im = ax.imshow(ref.T, origin="lower", cmap="viridis")
        ax.set_title(f"{k} (A)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[1, i]
        dm = np.where(both, d, np.nan)
        vmax = np.nanpercentile(np.abs(dm), 99) if np.isfinite(dm).any() else 1.0
        im = ax.imshow(dm.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{k}: B - A (99% range)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"\ndifference maps written to {out_png}")
