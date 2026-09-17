"""
Work-precision plots for the integrator sweep produced by src/tests/integrator_sweep.cpp.

Usage:
  python python/integrator_sweep_plot.py dat/integrator_sweep_lamppost.csv [dat/integrator_sweep_lamppost.png]

Left panels: error at the disc (90th percentile of |dg| and median |dr| against the reference
solution) versus wall-clock time; right panel: the same against mean steps per ray (machine
independent).  Each method is one line through its parameter sweep; symplectic runs are split by
order and far-field policy ("grow": max_tstep cap grows with r beyond maxtstep_rlim).  The lower
left of each panel is better.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", ".png")

rows = list(csv.DictReader(open(path)))
for r in rows:
    for k in r:
        if k != "method":
            r[k] = float(r[k])

def series():
    groups = {}
    for r in rows:
        if r["method"] in ("euler", "rk4", "rk45"):
            key = r["method"]
        else:
            key = f"{r['method']} o{int(r['order'])} h0={r['p1']:g}"
        groups.setdefault(key, []).append(r)
    return groups

groups = series()
style = {"euler": ("k", "x"), "rk4": ("tab:gray", "s"), "rk45": ("tab:red", "o")}
cmap4 = plt.get_cmap("Blues"); cmap6 = plt.get_cmap("Greens")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
metrics = [("wall_s", "dg_p90", "wall time [s]", "90th percentile |dg| at disc"),
           ("wall_s", "dr_med", "wall time [s]", "median |dr| at disc"),
           ("mean_steps", "dg_p90", "mean steps per ray", "90th percentile |dg| at disc")]
keys = sorted(groups)
n4 = sum(1 for k in keys if " o4 " in k); n6 = sum(1 for k in keys if " o6 " in k)
i4 = i6 = 0
for key in keys:
    g = groups[key]
    if key in style:
        c, m = style[key]; ls = "-"
    elif " o4 " in key:
        c = cmap4(0.35 + 0.6 * i4 / max(n4 - 1, 1)); i4 += 1; m = "^" if "grow" in key else "v"; ls = "--" if "grow" in key else "-"
    else:
        c = cmap6(0.35 + 0.6 * i6 / max(n6 - 1, 1)); i6 += 1; m = "^" if "grow" in key else "v"; ls = "--" if "grow" in key else "-"
    for ax, (xk, yk, xl, yl) in zip(axes, metrics):
        x = np.array([r[xk] for r in g]); y = np.array([r[yk] for r in g])
        o = np.argsort(x)
        ax.plot(x[o], np.maximum(y[o], 1e-16), ls, marker=m, color=c, label=key, ms=5, lw=1.2)
for ax, (xk, yk, xl, yl) in zip(axes, metrics):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, which="both", alpha=0.3)
axes[0].set_title(path.split("/")[-1].replace(".csv", ""))
axes[2].legend(fontsize=7, loc="upper right", ncol=2)
plt.tight_layout()
plt.savefig(out, dpi=130)
print("written", out)
