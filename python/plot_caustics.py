import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

with fits.open('dat/caustic_sourceplane_test.fits') as f:
    hdr   = f['DET_J'].header
    det_j = f['DET_J'].data
    sign_j = f['SIGN_J'].data
    order  = f['ORDER'].data
    escaped = f['ESCAPED'].data
    theta_s = f['THETA_S'].data
    phi_s   = f['PHI_S'].data

x0   = hdr['X0'];   xmax = hdr['XMAX']
nx   = hdr['NX'];   ny   = hdr['NY']
x = np.linspace(x0, xmax, nx)
y = np.linspace(hdr['Y0'], hdr['YMAX'], ny)

print('ORDER distribution (escaped rays):')
esc = escaped == 1
for o in range(-1, 5):
    n = np.sum(order[esc] == o)
    if n > 0:
        print('  order=%d: %6d pixels (%.1f%%)' % (o, n, 100*n/np.sum(esc)))

mask_outer = (np.sqrt(x[:,None]**2 + y[None,:]**2) > 8) & esc & np.isfinite(sign_j)
print()
print('Outer region (r>8 rg, escaped): sign=+1: %d, sign=-1: %d' % (
    np.sum(sign_j[mask_outer]>0), np.sum(sign_j[mask_outer]<0)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
zoom = 10

def do_zoom(ax, data, cmap, vmin, vmax, title):
    ix0 = np.searchsorted(x, -zoom); ix1 = np.searchsorted(x, zoom)
    iy0 = np.searchsorted(y, -zoom); iy1 = np.searchsorted(y, zoom)
    im = ax.pcolormesh(x[ix0:ix1], y[iy0:iy1], data[ix0:ix1, iy0:iy1],
                       cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_aspect('equal')
    ax.set_xlabel('x (rg)'); ax.set_ylabel('y (rg)')
    plt.colorbar(im, ax=ax)

do_zoom(axes[0], sign_j, 'bwr', -1, 1, 'SIGN_J (fixed order)')
order_f = order.astype(float); order_f[~esc] = np.nan
do_zoom(axes[1], order_f, 'tab10', -0.5, 4.5, 'ORDER (corrected)')
do_zoom(axes[2], escaped.astype(float), 'Greys_r', 0, 1, 'ESCAPED')

plt.tight_layout()
plt.savefig('dat/caustic_sourceplane_fixed.png', dpi=130)
print('Saved dat/caustic_sourceplane_fixed.png')