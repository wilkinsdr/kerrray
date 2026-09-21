"""
Continuum and flux-cube images produced by src/ray_transfer/ray_transfer_disc_surface.cpp.

Usage:
  python python/ray_transfer_disc_surface_plot.py dat/ray_transfer_disc_surface.fits [dat/ray_transfer_disc_surface.png] [--show]

Plots the CONTINUUM extension (disc's emitting annulus) and one FLUX-cube energy slice near the line's
rest energy (the middle frame of the cube) side by side. Pass --show to also display the figure on screen.
"""
import sys

show = "--show" in sys.argv
if show:
    sys.argv.remove("--show")

from astropy.io import fits
import numpy as np
import matplotlib
if not show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".fits", ".png")

hdul = fits.open(path)
continuum = hdul["CONTINUUM"].data
ne = hdul[0].header.get("NE")
n_ext = len(hdul)
# the flux cube occupies extensions 3..n_ext-1 (CONTINUUM is 1, TAU is 2); take the middle energy slice
mid_ext = 3 + (n_ext - 3) // 2
flux_mid = hdul[mid_ext].data

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
im0 = axes[0].imshow(continuum, origin="lower", cmap="inferno")
axes[0].set_title("Continuum (disc annulus only)")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(flux_mid, origin="lower", cmap="inferno")
axes[1].set_title("Flux (mid energy bin)")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
if show:
    plt.show()
