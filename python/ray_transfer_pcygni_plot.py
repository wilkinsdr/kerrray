"""
P-Cygni residual flux profile produced by src/ray_transfer/ray_transfer_pcygni.cpp.

Usage:
  python python/ray_transfer_pcygni_plot.py dat/ray_transfer_pcygni.csv [dat/ray_transfer_pcygni.png]
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", ".png")

rows = list(csv.DictReader(open(path)))
energy = [float(r["energy"]) for r in rows]
residual = [float(r["residual"]) for r in rows]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(energy, residual, color="C0")
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("energy")
ax.set_ylabel("residual flux (continuum = 1)")
ax.set_title("P-Cygni profile (FlatRayTransfer, spherical wind, no black hole)")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
