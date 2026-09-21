"""
Wind line-emission spectrum for the R0 = 200, kappa0 tuned so peak wind optical depth = 1 run
(docs/plan_ray_transfer.md Sec 5.18). No pixel reached the corona at this field of view/resolution, so
continuum = 0 everywhere and total_flux == line_flux; this plots line_flux directly (not the usual
continuum-normalised residual, which is undefined here).

Usage:
  python python/ray_transfer_disc_wind_R200_tau1_plot.py dat/ray_transfer_disc_wind_R200_tau1_spectrum.csv [out.png]
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "dat/ray_transfer_disc_wind_R200_tau1_spectrum.csv"
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", ".png")

rows = list(csv.DictReader(open(path)))
energy = [float(r["energy"]) for r in rows]
line = [float(r["line_flux"]) for r in rows]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(energy, line, color="C0")
ax.axvline(6.4, color="gray", lw=0.8, ls="--", label="rest energy")
ax.set_xlabel("energy (keV)")
ax.set_ylabel("line flux (no continuum sampled)")
ax.set_title("R0 = 200, peak wind tau = 1: wind emission line")
ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
