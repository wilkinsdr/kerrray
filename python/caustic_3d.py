"""
caustic_3d.py

Plots the 3D caustic map produced by caustic_3d (CAUSTICS table + per-pixel maps).

Figure 1 (<out>_3d.png/.pdf): 3D scatter of the caustic points, coloured by crossing index N
    (1 = first caustic along the ray, 2 = second, ...), with the horizon sphere and the observer
    direction for reference.
Figure 2 (<out>_projections.png/.pdf):
    (a) projection onto the plane containing the line of sight and the image-plane x axis
        (Z' = distance towards the observer, X' = image-plane x direction)
    (b) projection onto the plane containing the line of sight and the image-plane y axis
    (c) cross-section of the caustic surface in a plane perpendicular to the line of sight
        (points with Z' within +-dz of a chosen value; default: the slab behind the black hole
        containing the most points), showing the astroid shape of the tube
    (d) intersection with the equatorial plane (points with |cos theta| < 0.05), i.e. the disc caustics
    (e) sky map of the number of caustic crossings per ray (NCAUST)
    (f) sky map of the ray fate (STATUS)

Coordinates: the FITS table gives Boyer-Lindquist (r, theta, phi) and the Cartesian (X, Y, Z) of
kerr.h cartesian().  The observer frame used here is
    n  = (sin i cos phi0, sin i sin phi0, cos i)        line of sight, towards the observer  (Z')
    ex = (-sin phi0, cos phi0, 0)                       image-plane x                          (X')
    ey = (-cos i cos phi0, -cos i sin phi0, sin i)      image-plane y (towards the pole)       (Y')

Usage (from project root):
  python python/caustic_3d.py [fits_file] [output_prefix] [--nmax N] [--slab Z'] [--dz DZ] [--zoom W] [--rmax R]

Defaults: fits_file = dat/caustic_3d.fits, output_prefix = dat/caustic_3d
"""

import sys
import argparse
import numpy as np
import astropy.io.fits as pyfits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fits_file", nargs="?", default="dat/caustic_3d.fits")
ap.add_argument("out_prefix", nargs="?", default="dat/caustic_3d")
ap.add_argument("--nmax", type=int, default=4, help="highest crossing index to plot (default 4)")
ap.add_argument("--slab", type=float, default=None, help="Z' of the cross-section slab (default: densest slab behind the hole)")
ap.add_argument("--dz", type=float, default=0.1, help="half-thickness of the cross-section slab (rg)")
ap.add_argument("--rmax", type=float, default=None, help="only plot points with r < rmax (default: 3 x outer image-plane extent)")
ap.add_argument("--zoom", type=float, default=0.05, help="minimum half-width of the cross-section panel (rg)")
ap.add_argument("--no-show", action="store_true")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with pyfits.open(args.fits_file) as f:
    hdr = f[0].header
    spin = hdr["SPIN"]; incl = hdr["INCL"]; phi0 = hdr.get("PHI0", 0.0)
    r_hor = hdr["HORIZON"]; r_isco = hdr["ISCO"]
    x0 = hdr["X0"]; xmax = hdr["XMAX"]; y0 = hdr["Y0"]; ymax = hdr["YMAX"]
    t = f["CAUSTICS"].data
    ncaust = np.array(f["NCAUST"].data)
    status = np.array(f["STATUS"].data)

i = np.radians(incl)
n_hat = np.array([np.sin(i) * np.cos(phi0), np.sin(i) * np.sin(phi0), np.cos(i)])
e_x = np.array([-np.sin(phi0), np.cos(phi0), 0.0])
e_y = np.array([-np.cos(i) * np.cos(phi0), -np.cos(i) * np.sin(phi0), np.sin(i)])

X = np.vstack([t["X"], t["Y"], t["Z"]]).T
Xp = X @ e_x
Yp = X @ e_y
Zp = X @ n_hat
N = np.array(t["N"])
r = np.array(t["R"])
theta = np.array(t["THETA"])

rmax = args.rmax if args.rmax is not None else 3 * max(abs(x0), abs(xmax), abs(y0), abs(ymax))
sel = (r < rmax) & (N <= args.nmax)
print(f"Loaded {args.fits_file}: {len(t)} caustic points, {sel.sum()} plotted (r < {rmax:g}, N <= {args.nmax})")
print(f"spin = {spin}, incl = {incl} deg, horizon = {r_hor:.3f}, ISCO = {r_isco:.3f}")
for k in range(1, args.nmax + 1):
    m = sel & (N == k)
    if m.sum():
        print(f"  N = {k}: {m.sum()} points, r in [{r[m].min():.2f}, {r[m].max():.2f}]")

# ---------------------------------------------------------------------------
# Styling: one sequential ramp for the crossing index, muted ink for everything else
# ---------------------------------------------------------------------------
ramp = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, args.nmax))
cmap = ListedColormap(ramp)
norm = BoundaryNorm(np.arange(0.5, args.nmax + 1.5), cmap.N)
ink = "#333333"
muted = "#888888"
plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                     "axes.titlecolor": ink, "font.size": 9})


def add_colorbar(fig, ax, sc, label="crossing index N"):
    cb = fig.colorbar(sc, ax=ax, ticks=np.arange(1, args.nmax + 1), fraction=0.046, pad=0.04)
    cb.set_label(label)
    return cb


def horizon_circle(ax, plane="xz"):
    ph = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_hor * np.cos(ph), r_hor * np.sin(ph), color=ink, lw=0.8)
    ax.plot(r_isco * np.cos(ph), r_isco * np.sin(ph), color=muted, lw=0.6, ls="--")


# ---------------------------------------------------------------------------
# Figure 1: 3D scatter
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
order = np.argsort(-N[sel])   # draw high N first so the primary caustic is on top
sc = ax.scatter(Xp[sel][order], Yp[sel][order], Zp[sel][order], c=N[sel][order], cmap=cmap, norm=norm,
                s=1.5, alpha=0.6, linewidths=0)
# horizon sphere
u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
ax.plot_surface(r_hor * np.cos(u) * np.sin(v), r_hor * np.sin(u) * np.sin(v), r_hor * np.cos(v),
                color="#bbbbbb", alpha=0.5, linewidth=0)
# spin axis (in the observer frame the BH spin axis is (0, sin i, cos i))
L = rmax * 0.4
ax.plot([0, 0], [0, L * np.sin(i)], [0, L * np.cos(i)], color=ink, lw=1.2)
ax.text(0, L * np.sin(i), L * np.cos(i), " spin axis", color=ink, fontsize=8)
ax.plot([0, 0], [0, 0], [0, L], color=muted, lw=1.0, ls="--")
ax.text(0, 0, L, " to observer", color=muted, fontsize=8)
lim = rmax
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_xlabel("X' (image x) [rg]"); ax.set_ylabel("Y' (image y) [rg]"); ax.set_zlabel("Z' (towards observer) [rg]")
ax.set_box_aspect((1, 1, 1))
ax.set_title(f"Caustics: a = {spin}, i = {incl:g}°", loc="left")
add_colorbar(fig, ax, sc)
fig.savefig(args.out_prefix + "_3d.png", dpi=200, bbox_inches="tight")
fig.savefig(args.out_prefix + "_3d.pdf", bbox_inches="tight")
print("Wrote", args.out_prefix + "_3d.png/.pdf")

# ---------------------------------------------------------------------------
# Figure 2: projections, cross-section, disc intersection, sky maps
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
axs = axs.ravel()

# (a) line of sight vs image x
ax = axs[0]
ax.scatter(Zp[sel][order], Xp[sel][order], c=N[sel][order], cmap=cmap, norm=norm, s=1, alpha=0.6, linewidths=0)
horizon_circle(ax)
ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax / 2, rmax / 2)
ax.set_aspect("equal")
ax.set_xlabel("Z' (towards observer) [rg]"); ax.set_ylabel("X' (image x) [rg]")
ax.set_title("(a) projection: line of sight vs image x", loc="left")

# (b) line of sight vs image y
ax = axs[1]
ax.scatter(Zp[sel][order], Yp[sel][order], c=N[sel][order], cmap=cmap, norm=norm, s=1, alpha=0.6, linewidths=0)
horizon_circle(ax)
ax.plot([0, -L * np.sin(i)], [0, -L * np.cos(i)], color=muted, lw=0.8)   # spin axis projection (south)
ax.plot([0, L * np.sin(i)], [0, L * np.cos(i)], color=ink, lw=0.8)       # spin axis projection (north)
ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax / 2, rmax / 2)
ax.set_aspect("equal")
ax.set_xlabel("Z' (towards observer) [rg]"); ax.set_ylabel("Y' (image y) [rg]")
ax.set_title("(b) projection: line of sight vs image y", loc="left")

# (c) cross-section perpendicular to the line of sight
ax = axs[2]
m1 = sel & (N == 1) & (Zp < 0)
if args.slab is None:
    if m1.sum():
        hist, edges = np.histogram(Zp[m1], bins=np.arange(-rmax, 0, 2 * args.dz))
        zslab = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    else:
        zslab = -5.0
else:
    zslab = args.slab
ms = sel & (np.abs(Zp - zslab) < args.dz)
sc = ax.scatter(Xp[ms], Yp[ms], c=N[ms], cmap=cmap, norm=norm, s=6, alpha=0.8, linewidths=0)
ax.set_aspect("equal")
ax.set_xlabel("X' (image x) [rg]"); ax.set_ylabel("Y' (image y) [rg]")
ax.set_title(f"(c) cross-section at Z' = {zslab:.2f} ± {args.dz:g} rg", loc="left")
mp = ms & (N == 1)
if mp.sum() >= 3:
    # zoom on the primary caustic tube: median of the N = 1 points in the slab +- max(zoom, 3 MAD)
    cx, cy = np.median(Xp[mp]), np.median(Yp[mp])
    rad = np.hypot(Xp[mp] - cx, Yp[mp] - cy)
    w = max(args.zoom, 1.5 * np.percentile(rad, 98))
    ax.set_xlim(cx - w, cx + w); ax.set_ylim(cy - w, cy + w)
elif ms.sum():
    w = max(np.ptp(Xp[ms]), np.ptp(Yp[ms]), 0.5) * 0.75
    cx, cy = np.median(Xp[ms]), np.median(Yp[ms])
    ax.set_xlim(cx - w, cx + w); ax.set_ylim(cy - w, cy + w)

# (d) equatorial plane
ax = axs[3]
me = sel & (np.abs(np.cos(theta)) < 0.05)
xe = np.sqrt(r[me] ** 2 + spin ** 2) * np.cos(t["PHI"][me])
ye = np.sqrt(r[me] ** 2 + spin ** 2) * np.sin(t["PHI"][me])
ax.scatter(xe, ye, c=N[me], cmap=cmap, norm=norm, s=3, alpha=0.8, linewidths=0)
horizon_circle(ax)
ax.plot([0, rmax * np.cos(phi0)], [0, rmax * np.sin(phi0)], color=muted, lw=0.8, ls=":")
ax.text(rmax * 0.55 * np.cos(phi0), rmax * 0.55 * np.sin(phi0), "to observer", color=muted, fontsize=8)
ax.set_xlim(-rmax / 2, rmax / 2); ax.set_ylim(-rmax / 2, rmax / 2)
ax.set_aspect("equal")
ax.set_xlabel("x [rg]"); ax.set_ylabel("y [rg]")
ax.set_title("(d) intersection with the equatorial plane", loc="left")

# (e) sky map: crossings per ray
ax = axs[4]
ext = [x0, xmax, y0, ymax]
im = ax.imshow(ncaust, origin="lower", extent=ext,   # FITS image arrays are indexed [iy, ix]
               cmap=plt.get_cmap("viridis", int(ncaust.max()) + 1), vmin=-0.5, vmax=ncaust.max() + 0.5,
               interpolation="nearest")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="caustic crossings along the ray")
ax.set_xlabel("image x [rg]"); ax.set_ylabel("image y [rg]")
ax.set_title("(e) sky map: number of caustic crossings", loc="left")

# (f) sky map: ray fate
ax = axs[5]
fate_cmap = ListedColormap(["#d9d9d9", "#333333", "#c44e52", "#dd8452", "#8172b3", "#937860"])
im = ax.imshow(status, origin="lower", extent=ext,
               cmap=fate_cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(6))
cb.ax.set_yticklabels(["escaped", "horizon", "step limit", "bundle split", "skipped", "NaN"])
ax.set_xlabel("image x [rg]"); ax.set_ylabel("image y [rg]")
ax.set_title("(f) sky map: ray fate", loc="left")

fig.suptitle(f"Kerr caustics for an observer at i = {incl:g}°, a = {spin}  (colour: crossing index N)", x=0.01,
             ha="left", color=ink)
fig.tight_layout()
fig.savefig(args.out_prefix + "_projections.png", dpi=200, bbox_inches="tight")
fig.savefig(args.out_prefix + "_projections.pdf", bbox_inches="tight")
print("Wrote", args.out_prefix + "_projections.png/.pdf")

if not args.no_show and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
    plt.show()
