"""
caustic_discplane.py

Plots the caustic / critical-curve analysis from caustic_discplane.fits in a
single 3x3 figure:

  Row 1: image order map | det(J) | critical curves on sky
  Row 2: zoom order map  | zoom det(J) | parity sign(det J)
  Row 3: disc radius     | disc phi    | redshift

Usage (from project root):
  python python/caustic_discplane.py [fits_file] [output_pdf]

Defaults:
  fits_file   = dat/caustic_discplane.fits
  output_pdf  = dat/caustic_discplane_summary.pdf
"""

import sys
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FITS_FILE = sys.argv[1] if len(sys.argv) > 1 else "dat/caustic_discplane.fits"
OUT_FILE  = sys.argv[2] if len(sys.argv) > 2 else "dat/caustic_discplane_summary.pdf"
SENTINEL  = 1e30
ZOOM_HALF = 8.0   # rg, for photon-ring zoom panels

# ---------------------------------------------------------------------------
# Load FITS
# ---------------------------------------------------------------------------
with pyfits.open(FITS_FILE) as f:
    spin = f[0].header['SPIN']
    incl = f[0].header['INCL']
    isco = f[0].header['ISCO']

    det_J    = np.array(f['DET_J'].data)
    sign_J   = np.array(f['SIGN_J'].data)
    order    = np.array(f['ORDER'].data)
    hit      = np.array(f['HIT'].data)
    radius   = np.array(f['RADIUS'].data)
    phi      = np.array(f['PHI'].data)
    redshift = np.array(f['REDSHIFT'].data)

    hdr  = f['DET_J'].header
    x0   = hdr['X0'];  xmax = hdr['XMAX']
    y0   = hdr['Y0'];  ymax = hdr['YMAX']

extent = [x0, xmax, y0, ymax]
Ny, Nx = det_J.shape
xvals  = np.linspace(x0, xmax, Nx)
yvals  = np.linspace(y0, ymax, Ny)

print(f"Loaded {FITS_FILE}")
print(f"  spin={spin}, incl={incl}°, ISCO={isco:.3f} rg")
print(f"  {Nx} × {Ny} pixels, extent [{x0}, {xmax}] × [{y0}, {ymax}] rg")

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
# Order display (NaN where no disc hit)
order_display = np.where(hit == 1, order.astype(float), np.nan)
max_order = int(np.nanmax(order_display))

cmap_order = plt.cm.get_cmap('tab10', max_order + 1)
norm_order = mcolors.BoundaryNorm(np.arange(-0.5, max_order + 1.5), max_order + 1)

# det_J plot (mask sentinels and no-hit)
valid = (hit == 1) & np.isfinite(det_J) & (np.abs(det_J) < SENTINEL * 0.5)
det_J_plot = np.where(valid, det_J, np.nan)
vmax_J = np.nanpercentile(np.abs(det_J_plot[valid]), 95)

# Image-order boundary mask
order_hit = np.where(hit == 1, order, -99)
order_boundary = np.zeros_like(order, dtype=bool)
order_boundary[1:,  :] |= order_hit[1:,  :] != order_hit[:-1, :]
order_boundary[:-1, :] |= order_hit[:-1, :] != order_hit[1:,  :]
order_boundary[:,  1:] |= order_hit[:,  1:] != order_hit[:, :-1]
order_boundary[:, :-1] |= order_hit[:, :-1] != order_hit[:,  1:]
order_boundary &= (hit == 1)

# Parity / sign display
sJ = sign_J.copy().astype(float)
sJ[~np.isfinite(det_J)] = np.nan
sJ[np.abs(det_J) >= SENTINEL * 0.5] = np.nan
sJ[hit == 0] = np.nan

# Disc maps (masked)
radius_plot   = np.where(hit == 1, radius,   np.nan)
phi_plot      = np.where(hit == 1, phi,      np.nan)
redshift_plot = np.where(hit == 1, redshift, np.nan)

# ---------------------------------------------------------------------------
# Figure layout: 3 rows × 3 columns
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 17))
fig.suptitle(
    f"Kerr caustic structure  |  spin = {spin},  incl = {incl}°,  ISCO = {isco:.3f} r$_g$",
    fontsize=14, y=0.995
)

kw_imshow = dict(origin='lower', interpolation='nearest')

def label_axes(ax, xlabel=True, ylabel=True):
    if xlabel:
        ax.set_xlabel("$x$ (r$_g$)", fontsize=10)
    if ylabel:
        ax.set_ylabel("$y$ (r$_g$)", fontsize=10)

def overlay_cc(ax, lw=0.8, color='white', ls='-', alpha=1.0):
    """Overlay det(J)=0 critical curve contour."""
    ax.contour(xvals, yvals, det_J_plot, levels=[0],
               colors=color, linewidths=lw, linestyles=ls, alpha=alpha)

# ---- (0,0) Image order map ------------------------------------------------
ax = axes[0, 0]
im = ax.imshow(order_display, extent=extent, cmap=cmap_order, norm=norm_order,
               **kw_imshow)
cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, max_order + 1), fraction=0.046, pad=0.04)
cbar.set_label("Image order $n$", fontsize=9)
ax.set_title("Image order map", fontsize=11)
label_axes(ax)

# ---- (0,1) det(J) map -----------------------------------------------------
ax = axes[0, 1]
im = ax.imshow(det_J_plot, extent=extent, cmap='RdBu_r',
               norm=mcolors.Normalize(vmin=-vmax_J, vmax=vmax_J), **kw_imshow)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\det J$", fontsize=9)
ax.set_title(r"Jacobian determinant $\det J$", fontsize=11)
label_axes(ax)

# ---- (0,2) Critical curves on sky ----------------------------------------
ax = axes[0, 2]
ax.imshow(order_display, extent=extent, cmap=cmap_order, norm=norm_order,
          alpha=0.7, **kw_imshow)
overlay_cc(ax, lw=1.2)
ob_display = np.where(order_boundary, 1.0, np.nan)
ax.imshow(ob_display, extent=extent, cmap='autumn', vmin=0, vmax=1, alpha=0.9,
          **kw_imshow)
legend_elems = [
    Line2D([0], [0], color='white', lw=1.5,
           label=r"$\det J = 0$ critical curve"),
    Patch(facecolor='red', alpha=0.8, label="Order boundary"),
]
ax.legend(handles=legend_elems, loc='upper right', fontsize=7,
          framealpha=0.7, facecolor='0.2', labelcolor='white')
ax.set_title("Critical curves on sky", fontsize=11)
label_axes(ax)

# ---- (1,0) Zoom: order map ------------------------------------------------
ax = axes[1, 0]
im = ax.imshow(order_display, extent=extent, cmap=cmap_order, norm=norm_order,
               **kw_imshow)
overlay_cc(ax, lw=0.8)
ax.set_xlim(-ZOOM_HALF, ZOOM_HALF)
ax.set_ylim(-ZOOM_HALF, ZOOM_HALF)
fig.colorbar(im, ax=ax, ticks=np.arange(0, max_order + 1), fraction=0.046, pad=0.04)
ax.set_title(f"Order map (zoom ±{ZOOM_HALF} r$_g$)", fontsize=11)
label_axes(ax)

# ---- (1,1) Zoom: det(J) ---------------------------------------------------
ax = axes[1, 1]
im = ax.imshow(det_J_plot, extent=extent, cmap='RdBu_r',
               norm=mcolors.Normalize(vmin=-vmax_J, vmax=vmax_J), **kw_imshow)
overlay_cc(ax, lw=0.8)
ax.set_xlim(-ZOOM_HALF, ZOOM_HALF)
ax.set_ylim(-ZOOM_HALF, ZOOM_HALF)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(r"$\det J$ (zoom)", fontsize=11)
label_axes(ax)

# ---- (1,2) Parity: sign(det J) --------------------------------------------
ax = axes[1, 2]
im = ax.imshow(sJ, extent=extent, cmap='coolwarm',
               norm=mcolors.Normalize(vmin=-1.2, vmax=1.2), **kw_imshow)
ax.contour(xvals, yvals, det_J_plot, levels=[0],
           colors='black', linewidths=1.0, linestyles='-')
cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
cbar.set_label("sign($\\det J$)", fontsize=9)
ax.set_title("Image parity  (black = critical curves)", fontsize=11)
label_axes(ax)

# ---- (2,0) Disc radius ----------------------------------------------------
ax = axes[2, 0]
im = ax.imshow(radius_plot, extent=extent,
               norm=mcolors.LogNorm(vmin=isco, vmax=50), cmap='viridis',
               **kw_imshow)
overlay_cc(ax, color='white', lw=0.7, ls='--', alpha=0.6)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("$r$ (r$_g$)", fontsize=9)
ax.set_title("Disc emission radius", fontsize=11)
label_axes(ax)

# ---- (2,1) Disc phi -------------------------------------------------------
ax = axes[2, 1]
im = ax.imshow(phi_plot, extent=extent,
               norm=mcolors.Normalize(vmin=-np.pi, vmax=np.pi), cmap='twilight',
               **kw_imshow)
overlay_cc(ax, color='white', lw=0.7, ls='--', alpha=0.6)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("$\\phi$ (rad)", fontsize=9)
ax.set_title("Disc azimuthal angle", fontsize=11)
label_axes(ax)

# ---- (2,2) Redshift -------------------------------------------------------
ax = axes[2, 2]
im = ax.imshow(redshift_plot, extent=extent,
               norm=mcolors.Normalize(vmin=0.2, vmax=1.5), cmap='RdYlBu_r',
               **kw_imshow)
overlay_cc(ax, color='white', lw=0.7, ls='--', alpha=0.6)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("$E_{\\rm obs}/E_{\\rm emit}$", fontsize=9)
ax.set_title("Redshift $E_{\\rm obs}/E_{\\rm emit}$", fontsize=11)
label_axes(ax)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(OUT_FILE, bbox_inches='tight', dpi=150)
print(f"Saved: {OUT_FILE}")
plt.show()
