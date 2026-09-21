"""
Kerr vs flat-space comparison produced by src/ray_transfer/ray_transfer_kerr_vs_flat.cpp -- the same
spherical wind + static corona/star traced through the full Kerr (Schwarzschild, by default) spacetime and
through flat spacetime, isolating the effect of gravitational redshift.

Usage:
  python python/ray_transfer_kerr_vs_flat_plot.py dat/ray_transfer_kerr_vs_flat.csv [out.png] [--show]
"""
import sys

show = "--show" in sys.argv
if show:
    sys.argv.remove("--show")

import csv
import matplotlib
if not show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", ".png")

rows = list(csv.DictReader(open(path)))
energy = [float(r["energy"]) for r in rows]
kerr = [float(r["kerr_residual"]) for r in rows]
flat = [float(r["flat_residual"]) for r in rows]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(energy, flat, color="C0", label="flat (Minkowski)")
ax.plot(energy, kerr, color="C1", label="Kerr (Schwarzschild)")
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("energy")
ax.set_ylabel("flux / unabsorbed continuum")
ax.set_title("Same wind + corona: flat spacetime vs Schwarzschild\n(difference = gravitational redshift)")
ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
if show:
    plt.show()
