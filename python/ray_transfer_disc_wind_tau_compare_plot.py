"""
Overlay of the R0 = 50 disc-wind spectrum at several values of peak wind optical depth to the corona
(docs/plan_ray_transfer.md Sec 5.19-5.21).

Usage:
  python python/ray_transfer_disc_wind_tau_compare_plot.py out.png label1:path1.csv label2:path2.csv ...
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
entries = [arg.split(":", 1) for arg in sys.argv[2:]]


def load(path):
    rows = list(csv.DictReader(open(path)))
    energy = [float(r["energy"]) for r in rows]
    residual = [float(r["residual"]) for r in rows]
    return energy, residual


fig, ax = plt.subplots(figsize=(6.5, 4.2))
colors = ["C0", "C1", "C3", "C2", "C4"]
for (label, path), color in zip(entries, colors):
    e, r = load(path)
    ax.plot(e, r, color=color, label=label)
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.axvline(6.4, color="gray", lw=0.6, ls=":")
ax.set_xlabel("energy (keV)")
ax.set_ylabel("flux / unabsorbed continuum")
ax.set_title("R0 = 50 disc-wind spectrum comparison")
ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
