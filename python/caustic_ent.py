"""
caustic_ent.py

Plots the caustic / disc-plane intersection curves produced by caustic_ent in the energy/time plane of the
reverberation response function ("ent plot"), on the disc plane and on the image plane.

Figure (<out>.png/.pdf), three panels:
    (a) energy/time: the caustic curves (ENERGY = line energy / g, TIME as in the ent code) over-plotted on
        the RESPONSE image of a disc_ent_line_rbin FITS file — the one given on the command line, or else the
        one recorded in the ENTFILE keyword of the caustic_ent file (i.e. the file caustic_ent took the
        source-to-disc times from, when run with --entfile=) — otherwise on the density of disc pixels in the
        same plane (from the GDISC<n>/TDISC<n> maps and the DISC table of the caustic_ent file), or on nothing
        with --no-background
    (b) the disc plane (X, Y) with the same curves, the horizon, the ISCO and the disc edges; the observer is
        towards +X' (the image-plane azimuth phi0)
    (c) the image plane with the curves over the map of equatorial crossings per ray

Curves are drawn per sheet (equatorial crossing order n of the rays) and labelled with the order N of the
caustic meeting the disc there (NCAUST); vertices with ACCEPT = 0 (crossing position discontinuous) or with
undefined energy (inside the ISCO / outside the disc) break the line.

Usage (from project root):
  python python/caustic_ent.py [caustic_ent.fits] [ent.fits] [--out prefix] [--sheets 1,2] [--tmin T --tmax T]
                               [--emin E --emax E] [--rlim R] [--all] [--no-background] [--no-show]

Defaults: caustic_ent.fits = dat/caustic_ent.fits, output prefix = the caustic file without extension.
"""

import sys
import argparse
import numpy as np
import astropy.io.fits as pyfits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fits_file", nargs="?", default="dat/caustic_ent.fits", help="caustic_ent output")
ap.add_argument("ent_file", nargs="?", default=None,
                help="disc_ent_line_rbin output to over-plot on (default: the ENTFILE keyword of the caustic file)")
ap.add_argument("--no-background", action="store_true", help="draw the curves alone in panel (a)")
ap.add_argument("--out", default=None, help="output prefix (default: the caustic file without extension)")
ap.add_argument("--sheets", default=None, help="comma-separated sheets (equatorial crossing orders) to plot")
ap.add_argument("--tmin", type=float, default=None)
ap.add_argument("--tmax", type=float, default=None)
ap.add_argument("--emin", type=float, default=None)
ap.add_argument("--emax", type=float, default=None)
ap.add_argument("--rlim", type=float, default=None, help="half-width of the disc-plane panel (rg; default: fits the curves)")
ap.add_argument("--seg_jump", type=float, default=2.0, help="break a curve where consecutive vertices jump by more than this (rg) + 20%% of r")
ap.add_argument("--all", action="store_true", help="also draw vertices with ACCEPT = 0 (dotted)")
ap.add_argument("--no-show", action="store_true")
args = ap.parse_args()

out_prefix = args.out if args.out else args.fits_file.rsplit(".", 1)[0]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with pyfits.open(args.fits_file) as f:
    hdr = f[0].header
    spin = hdr["SPIN"]; incl = hdr["INCL"]; phi0 = hdr.get("PHI0", 0.0)
    r_hor = hdr["HORIZON"]; r_isco = hdr["ISCO"]; r_min = hdr["RMIN"]; r_disc = hdr["RDISC"]
    line_en = hdr["LINEEN"]; t_cont = hdr["TSTART"]; rgs = hdr["RGS"]; tunit = hdr.get("TUNIT", "GM/c^3")
    max_eq = hdr["MAXEQCR"]
    ent_from_hdr = str(hdr.get("ENTFILE", "")).strip()
    x0 = hdr["X0"]; xmax = hdr["XMAX"]; y0 = hdr["Y0"]; ymax = hdr["YMAX"]
    t = f["CURVES"].data
    neq = np.array(f["NEQCROSS"].data)
    pix = [(np.array(f[f"GDISC{n}"].data), np.array(f[f"TDISC{n}"].data), np.array(f[f"RDISC{n}"].data))
           for n in range(1, max_eq + 1)]
    disc = f["DISC"].data
    refined = np.array(f["POINTS"].data) if "POINTS" in f else None

sheets = sorted(set(int(s) for s in t["SHEET"]))
if args.sheets:
    sheets = [int(s) for s in args.sheets.split(",")]
print(f"Loaded {args.fits_file}: {len(t)} vertices on sheets {sheets}; spin = {spin}, incl = {incl} deg, "
      f"line energy {line_en}, t_continuum = {t_cont}, times in {tunit}")
for n in sheets:
    m = t["SHEET"] == n
    if m.sum():
        acc = m & (t["ACCEPT"] == 1)
        print(f"  sheet {n}: {len(set(t['CURVE'][m]))} curve(s), {m.sum()} vertices ({acc.sum()} accepted), "
              f"r in [{np.nanmin(t['R'][m]):.2f}, {np.nanmax(t['R'][m]):.2f}], caustic orders {sorted(set(t['NCAUST'][m]))}")

# direct-image (sheet 1) locus of the whole disc in the energy/time plane, for the background of panel (a)
disc_r = np.array(disc["r"]); disc_t = np.array(disc["time"])
ok = np.isfinite(disc_t)
logbin = disc_r.size > 2 and abs(disc_r[2] / disc_r[1] - disc_r[1] / disc_r[0]) < abs((disc_r[2] - disc_r[1]) - (disc_r[1] - disc_r[0]))
r_mid = disc_r * np.sqrt(disc_r[1] / disc_r[0]) if logbin else disc_r + 0.5 * (disc_r[1] - disc_r[0])
def t_sd(r):
    return np.interp(r, r_mid[ok], disc_t[ok], left=np.nan, right=np.nan) if ok.sum() > 1 else np.full_like(r, np.nan)
pix_E, pix_T = [], []
for g1, t1, r1 in pix:
    pm = np.isfinite(g1) & np.isfinite(t1) & (r1 >= r_min) & (r1 <= r_disc)
    E = line_en / g1[pm]; T = (t_sd(r1[pm]) + t1[pm] - t_cont) * rgs
    pm2 = np.isfinite(E) & np.isfinite(T)
    pix_E.append(E[pm2]); pix_T.append(T[pm2])
pix_E = np.concatenate(pix_E) if pix_E else np.array([]); pix_T = np.concatenate(pix_T) if pix_T else np.array([])

# ent response: the file named on the command line, or the one caustic_ent read its disc table from
if args.ent_file is None and ent_from_hdr and not args.no_background:
    import os
    candidates = [ent_from_hdr, os.path.join(os.path.dirname(args.fits_file), os.path.basename(ent_from_hdr)),
                  os.path.normpath(os.path.join(os.path.dirname(args.fits_file), ent_from_hdr))]
    for c in candidates:
        if os.path.isfile(c):
            args.ent_file = c
            print(f"Using the ent file recorded in the caustic file's header: {c}")
            break
    else:
        print(f"Note: ENTFILE = '{ent_from_hdr}' in the header was not found; pass the ent file on the command line")
ent = None
if args.ent_file and not args.no_background:
    with pyfits.open(args.ent_file) as f:
        h = f["RESPONSE"].header
        img = np.array(f["RESPONSE"].data, dtype=float)
        et0 = h["T0"]; edt = h["DT"]; ent_nt = h["NT"]
        en0 = h["EN0"]; den = h["DEN"]; nen = h["NEN"]; enlog = bool(h.get("ENLOG", False))
        ent_tunit = h.get("TUNIT", tunit); ent_tstart = h.get("TSTART", None)
    if ent_tstart is not None and abs(ent_tstart - t_cont) > 1e-9:
        print(f"WARNING: t_continuum differs: ent file {ent_tstart}, caustic file {t_cont}")
    ent = dict(img=img, t0=et0, dt=edt, nt=ent_nt, en0=en0, den=den, nen=nen, enlog=enlog, tunit=ent_tunit)
    print(f"Loaded {args.ent_file}: RESPONSE {img.shape}, t = {et0}..{et0 + (ent_nt - 1) * edt} ({ent_tunit}), "
          f"E = {en0}..{en0 * den ** (nen - 1) if enlog else en0 + (nen - 1) * den} ({'log' if enlog else 'linear'})")

# ---------------------------------------------------------------------------
# Styling (as caustic_3d.py): one sequential ramp for the caustic order, muted ink elsewhere
# ---------------------------------------------------------------------------
nmax = int(max(1, np.nanmax(t["NCAUST"]))) if len(t) else 1
ramp = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, max(nmax, 2)))
sheet_ls = {1: "-", 2: "-", 3: "--", 4: ":"}
ink = "#333333"
muted = "#888888"
plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                     "axes.titlecolor": ink, "font.size": 9})


def curve_segments(mask):
    """Yield (indices) runs of consecutive accepted vertices of one curve, in curve order."""
    idx = np.where(mask)[0]
    idx = idx[np.argsort(t["VERTEX"][idx])]
    if idx.size == 0:
        return []
    good = (t["ACCEPT"][idx] == 1) if not args.all else np.ones(idx.size, bool)
    # break the line at rejected vertices and where consecutive vertices jump across the disc plane by more
    # than seg_jump (rg) + 20 % of r (e.g. contour segments passing through the horizon)
    runs, cur = [], []
    prev = None
    for k, ok_ in zip(idx, good):
        jump = (prev is not None and np.hypot(t["X"][k] - t["X"][prev], t["Y"][k] - t["Y"][prev])
                > args.seg_jump + 0.2 * min(t["R"][k], t["R"][prev]))
        if ok_ and not jump:
            cur.append(k)
        else:
            if len(cur) > 1: runs.append(np.array(cur))
            cur = [k] if ok_ else []
        prev = k if ok_ else None
    if len(cur) > 1: runs.append(np.array(cur))
    if t["CLOSED"][idx[0]] and len(runs) == 1 and runs[0].size == idx.size:
        runs[0] = np.append(runs[0], runs[0][0])
    return runs


def draw_curves(ax, xcol, ycol, xf=lambda v: v, yf=lambda v: v, lw=1.4):
    for n in sheets:
        for c in sorted(set(t["CURVE"][t["SHEET"] == n])):
            m = (t["SHEET"] == n) & (t["CURVE"] == c)
            for run in curve_segments(m):
                x = xf(np.array(t[xcol][run], dtype=float)); y = yf(np.array(t[ycol][run], dtype=float))
                # colour by the caustic order along the run (constant per run in practice)
                N = int(np.round(np.median(t["NCAUST"][run])))
                ax.plot(x, y, sheet_ls.get(n, "-"), color=ramp[min(N, nmax) - 1], lw=lw,
                        solid_capstyle="round")
            if args.all:
                bad = m & (t["ACCEPT"] == 0)
                if bad.sum():
                    ax.plot(xf(np.array(t[xcol][bad], dtype=float)), yf(np.array(t[ycol][bad], dtype=float)),
                            ".", color=muted, ms=3)


def legend_handles():
    h = [Line2D([], [], color=ramp[k], lw=1.6, label=f"caustic N = {k + 1}") for k in range(nmax)]
    h += [Line2D([], [], color=ink, lw=1.2, ls=sheet_ls.get(n, "-"), label=f"sheet n = {n}") for n in sheets]
    return h


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.9, wspace=0.28)

# (a) energy / time
ax = axes[0]
if ent is not None:
    tt = ent["t0"] + ent["dt"] * np.arange(ent["nt"] + 1) - 0.5 * ent["dt"]
    if ent["enlog"]:
        ee = ent["en0"] * ent["den"] ** (np.arange(ent["nen"] + 1) - 0.5)
    else:
        ee = ent["en0"] + ent["den"] * (np.arange(ent["nen"] + 1) - 0.5)
    img = ent["img"]
    # the image is written with time along X and energy along Y (write_image(ent, Nen, Nt, transpose=True))
    if img.shape != (ent["nen"], ent["nt"]):
        img = img.T
    pos = img[img > 0]
    vmax = pos.max() if pos.size else 1
    vmin = max(pos.min() if pos.size else 1e-3 * vmax, 1e-4 * vmax)
    pc = ax.pcolormesh(tt, ee, np.where(img > 0, img, np.nan), cmap="Greys", norm=LogNorm(vmin=vmin, vmax=vmax),
                       shading="flat", rasterized=True)
    if ent["enlog"]:
        ax.set_yscale("log")
    cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.set_label("line response (photon rate)")
    ax.set_title("(a) caustic curves on the line response", loc="left")
elif args.no_background:
    ax.set_title("(a) caustic curves in the energy/time plane", loc="left")
else:
    if pix_T.size:
        ax.hist2d(pix_T, pix_E, bins=150, cmap="Greys", norm=LogNorm(), rasterized=True)
    ax.set_title("(a) caustic curves over the density of disc pixels (all sheets)", loc="left")
draw_curves(ax, "TIME", "ENERGY")
if ent is None:
    ok_v = (t["ACCEPT"] == 1) & np.isfinite(t["TIME"]) & np.isfinite(t["ENERGY"])
    if ok_v.sum():
        tl = [min(t["TIME"][ok_v].min(), pix_T.min() if pix_T.size else np.inf), max(t["TIME"][ok_v].max(), pix_T.max() if pix_T.size else -np.inf)]
        el = [min(t["ENERGY"][ok_v].min(), pix_E.min() if pix_E.size else np.inf), max(t["ENERGY"][ok_v].max(), pix_E.max() if pix_E.size else -np.inf)]
        ax.set_xlim(tl[0] - 0.03 * (tl[1] - tl[0]), tl[1] + 0.03 * (tl[1] - tl[0]))
        ax.set_ylim(el[0] - 0.03 * (el[1] - el[0]), el[1] + 0.03 * (el[1] - el[0]))
ax.set_xlabel(f"time after continuum ({tunit})")
ax.set_ylabel("observed line energy")
if args.tmin is not None or args.tmax is not None: ax.set_xlim(args.tmin, args.tmax)
if args.emin is not None or args.emax is not None: ax.set_ylim(args.emin, args.emax)
ax.legend(handles=legend_handles(), loc="upper right", fontsize=7, frameon=False)

# (b) disc plane; rotate so that the observer's azimuth phi0 is along +X'
ax = axes[1]
c0, s0 = np.cos(phi0), np.sin(phi0)
rot_x = lambda v: v  # placeholder to keep draw_curves signature simple
Xr = t["X"] * c0 + t["Y"] * s0
Yr = -t["X"] * s0 + t["Y"] * c0
t_rot = {"XR": Xr, "YR": Yr}
for n in sheets:
    for c in sorted(set(t["CURVE"][t["SHEET"] == n])):
        m = (t["SHEET"] == n) & (t["CURVE"] == c)
        for run in curve_segments(m):
            N = int(np.round(np.median(t["NCAUST"][run])))
            ax.plot(Xr[run], Yr[run], sheet_ls.get(n, "-"), color=ramp[min(N, nmax) - 1], lw=1.4)
        if args.all:
            bad = m & (t["ACCEPT"] == 0)
            if bad.sum(): ax.plot(Xr[bad], Yr[bad], ".", color=muted, ms=3)
ph = np.linspace(0, 2 * np.pi, 361)
ax.fill(r_hor * np.cos(ph), r_hor * np.sin(ph), color="#222222", zorder=0)
ax.plot(r_isco * np.cos(ph), r_isco * np.sin(ph), color=muted, lw=0.8, ls="--")
ax.plot(r_min * np.cos(ph), r_min * np.sin(ph), color=muted, lw=0.8)
rr = np.array(t["R"][t["ACCEPT"] == 1]) if len(t) else np.array([10.0])
lim = min(r_disc, 1.15 * (np.nanmax(rr) if rr.size else 10.0))
lim = max(lim, 3 * r_isco)
if args.rlim: lim = args.rlim
ax.annotate("", xy=(0.95 * lim, 0), xytext=(0.75 * lim, 0), arrowprops=dict(arrowstyle="->", color=muted))
ax.text(0.85 * lim, 0.03 * lim, "to observer", color=muted, ha="center", fontsize=7)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("X' (rg)"); ax.set_ylabel("Y' (rg)")
ax.set_title("(b) disc plane (dashed: ISCO)", loc="left")

# (c) image plane
ax = axes[2]
nx, ny = neq.shape[1], neq.shape[0]
# FITS images are (Ny, Nx) with X along columns; write_image put X along axis 1
ax.imshow(neq, origin="lower", extent=[x0, xmax, y0, ymax], cmap="Greys", vmin=0, vmax=max(1, max_eq), alpha=0.6,
          interpolation="nearest", rasterized=True)
if refined is not None and (refined["LEVEL"] > 0).sum():
    m = refined["LEVEL"] > 0
    ax.plot(refined["XIMG"][m], refined["YIMG"][m], ".", color=muted, ms=0.6, alpha=0.5, rasterized=True, zorder=1)
draw_curves(ax, "XIMG", "YIMG", lw=1.2)
ax.set_aspect("equal")
ax.set_xlabel("image plane x (rg)"); ax.set_ylabel("image plane y (rg)")
ax.set_title("(c) image plane (grey: crossings per ray" + ("; dots: refined points)" if refined is not None else ")"), loc="left")

fig.suptitle(f"caustic / disc-plane intersection: a = {spin}, i = {incl:g} deg, line {line_en}", x=0.05, ha="left",
             color=ink)
for ext in ("png", "pdf"):
    fig.savefig(f"{out_prefix}.{ext}", dpi=150)
print(f"Saved {out_prefix}.png/.pdf")
if not args.no_show:
    plt.show()
