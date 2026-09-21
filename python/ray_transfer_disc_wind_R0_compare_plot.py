"""
Overlay of the disc-wind line spectrum at R0=5 (baseline) and R0=50, to illustrate how moving the
wind's launch radius out narrows and deepens the absorption feature (docs/plan_ray_transfer.md Sec 5.15).

Usage:
  python python/ray_transfer_disc_wind_R0_compare_plot.py dat/ray_transfer_disc_wind_spectrum.csv \
      dat/ray_transfer_disc_wind_R050_spectrum.csv [out.png]
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
path_r05 = sys.argv[1]
path_r050 = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "dat/ray_transfer_disc_wind_R0_compare.png"


def load(path):
    rows = list(csv.DictReader(open(path)))
    energy = [float(r["energy"]) for r in rows]
    residual = [float(r["residual"]) for r in rows]
    return energy, residual


e5, r5 = load(path_r05)
e50, r50 = load(path_r050)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

axes[0].plot(e5, r5, color="C0")
axes[0].axhline(1.0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("energy")
axes[0].set_ylabel("flux / unabsorbed continuum")
axes[0].set_title("R0 = 5, R_out = 100")

axes[1].plot(e50, r50, color="C3")
axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("energy")
axes[1].set_title("R0 = 50, R_out = 1000")

fig.suptitle("Disc-wind line spectrum: launch radius R0 = 5 vs 50 (same 20x R_out/R0 ratio)")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
