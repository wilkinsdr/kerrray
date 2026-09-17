"""
plot_caustic_discplane.py

Visualise the output of caustic_discplane — Kerr BH critical curves in the
image plane and their corresponding caustics on the accretion disc.

Three panels:
  Left   — SIGN_J in the image plane: parity map with critical curves at ±1 boundaries.
  Centre — DISC PLANE: critical-curve positions plotted as (x_disc, y_disc), coloured
           by image order.  This directly shows the caustic structure on the disc.
  Right  — ORDER map in the image plane (direct=0, first photon ring=1, …).

Run from the project root:
    python3 python/plot_caustic_discplane.py [path/to/caustic_discplane.fits] [output.png]
Defaults: dat/caustic_discplane.fits  →  dat/caustic_discplane.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

# --- File paths ---
fits_file = sys.argv[1] if len(sys.argv) > 1 else 'dat/caustic_discplane.fits'
out_file  = sys.argv[2] if len(sys.argv) > 2 else fits_file.replace('.fits', '.png')

with fits.open(fits_file) as f:
    phdr    = f[0].header           # primary HDU: global simulation keywords
    hdr     = f['DET_J'].header     # DET_J extension: axis keywords
    det_j   = f['DET_J'].data
    sign_j  = f['SIGN_J'].data
    order   = f['ORDER'].data
    hit     = f['HIT'].data
    x_disc  = f['X_DISC'].data
    y_disc  = f['Y_DISC'].data

# Image-plane axes
x0   = hdr['X0'];   xmax = hdr['XMAX'];  nx = hdr['NX']
y0   = hdr['Y0'];   ymax = hdr['YMAX'];  ny = hdr['NY']
x_img = np.linspace(x0, xmax, nx)
y_img = np.linspace(y0, ymax, ny)

spin   = phdr.get('SPIN',  '?')
incl   = phdr.get('INCL',  '?')
r_disc = phdr.get('RDISC', '?')
r_isco = phdr.get('ISCO',  '?')

print(f'Loaded {fits_file}')
print(f'  spin={spin}, incl={incl}°, r_disc={r_disc} rg, ISCO={r_isco} rg')
print(f'  Image plane: {nx}x{ny}, x=[{x0},{xmax}], y=[{y0},{ymax}]')
print()

hit_mask = hit == 1

# Order statistics
print('ORDER distribution (rays that hit disc):')
for o in range(-1, 5):
    n = int(np.sum(order[hit_mask] == o))
    if n > 0:
        print(f'  order={o}: {n:7d} pixels ({100*n/hit_mask.sum():.1f}%)')
print()

# Parity statistics in outer region
r_img = np.sqrt(x_img[:, None]**2 + y_img[None, :]**2)
outer = (r_img > 8) & hit_mask & np.isfinite(sign_j)
print(f'Outer region (r_img > 8 rg, hit disc): '
      f'sign=+1: {int(np.sum(sign_j[outer]>0))}, '
      f'sign=-1: {int(np.sum(sign_j[outer]<0))}')
print()

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(f'Kerr BH caustic structure — accretion disc\n'
             f'spin={spin}, incl={incl}°, r_disc={r_disc} rg', fontsize=11)

zoom = min(abs(x0), abs(xmax), abs(y0), abs(ymax))

def img_zoom(ax, data, cmap, vmin, vmax, title, xlabel='x_img (rg)', ylabel='y_img (rg)'):
    """Plot a zoomed sub-region of an image-plane map."""
    ix0 = np.searchsorted(x_img, -zoom);  ix1 = np.searchsorted(x_img, zoom)
    iy0 = np.searchsorted(y_img, -zoom);  iy1 = np.searchsorted(y_img, zoom)
    # data shape is (nx, ny); pcolormesh expects (ny, nx) → transpose
    im = ax.pcolormesh(x_img[ix0:ix1], y_img[iy0:iy1],
                       data[ix0:ix1, iy0:iy1].T,
                       cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel 1: SIGN_J in image plane
sign_plot = np.where(hit_mask, sign_j, np.nan)
img_zoom(axes[0], sign_plot, 'bwr', -1, 1,
         'Image plane — parity sign(det J)\n(critical curves at sign change)')

# Panel 2: Disc plane caustics
# Scatter-plot (x_disc, y_disc) positions, coloured by image order.
# Overlay critical-curve pixels (near det_j = 0) as bright points.
ax = axes[1]
for o, color in zip([0, 1, 2], ['steelblue', 'darkorange', 'forestgreen']):
    mask = hit_mask & (order == o) & np.isfinite(sign_j)
    if not np.any(mask): continue
    xd = x_disc[mask]
    yd = y_disc[mask]
    sv = sign_j[mask]
    alpha = 0.15
    ax.scatter(xd[sv > 0], yd[sv > 0], s=0.3, c=color, alpha=alpha,
               linewidths=0, label=f'order={o}, +' if o == 0 else None)
    ax.scatter(xd[sv < 0], yd[sv < 0], s=0.3, c=color, alpha=alpha * 0.5,
               linewidths=0)

# Overlay critical-curve pixels (det_j ≈ 0) and order-boundary pixels
sentinel = hdr.get('SENTINL', 1e29)
crit_mask = (hit_mask & np.isfinite(det_j) &
             (np.abs(det_j) < 0.01 * np.nanpercentile(
                 np.abs(det_j[hit_mask & np.isfinite(det_j)]), 90)))
order_bnd = hit_mask & (det_j == sentinel)

for omask, col, label in [
    (crit_mask & (order == 0), 'red',    'fold caustic (order 0)'),
    (crit_mask & (order == 1), 'magenta','fold caustic (order 1)'),
    (order_bnd,                'black',  'order boundary'),
]:
    if np.any(omask):
        ax.scatter(x_disc[omask], y_disc[omask], s=1.5, c=col,
                   linewidths=0, label=label, zorder=5)

# Draw ISCO circle if available
if r_isco != '?':
    theta_c = np.linspace(0, 2*np.pi, 300)
    ax.plot(float(r_isco)*np.cos(theta_c), float(r_isco)*np.sin(theta_c),
            'k--', lw=0.8, alpha=0.5, label=f'ISCO ({float(r_isco):.2f} rg)')

ax.set_xlim(-zoom, zoom)
ax.set_ylim(-zoom, zoom)
ax.set_aspect('equal')
ax.set_xlabel('x_disc (rg)')
ax.set_ylabel('y_disc (rg)')
ax.set_title('Disc plane — caustic structure\n(critical curves mapped to disc)')
ax.legend(loc='upper right', fontsize=7, markerscale=4)

# Panel 3: ORDER in image plane
order_f = np.where(hit_mask, order.astype(float), np.nan)
cmap_order = matplotlib.colormaps['tab10'].resampled(5)
img_zoom(axes[2], order_f, cmap_order, -0.5, 4.5,
         'Image plane — image order\n(0=direct, 1=first ring, …)')

plt.tight_layout()
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f'Saved {out_file}')
