"""
Image-plane-integrated line/continuum spectrum produced by src/ray_transfer/ray_transfer_disc_wind.cpp
(the *_spectrum.csv sibling of the main FITS output).

Usage:
  python python/ray_transfer_disc_wind_spectrum_plot.py dat/ray_transfer_disc_wind_spectrum.csv [out.png] [--show]
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
residual = [float(r["residual"]) for r in rows]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(energy, residual, color="C0")
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("energy")
ax.set_ylabel("flux / unabsorbed continuum")
ax.set_title("Disc-wind line spectrum (RayTransfer, Kerr backend)")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
if show:
    plt.show()
