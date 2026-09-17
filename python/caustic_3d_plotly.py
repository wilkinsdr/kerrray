"""
caustic_3d_plotly.py

Interactive 3D view of the caustic map produced by caustic_3d, written as a self-contained HTML file.

The caustic points are drawn in the observer frame (Z' towards the observer, X' and Y' along the
image-plane axes -- see caustic_3d.py for the definition), one trace per crossing index N so that each
order can be toggled in the legend.  The event horizon, the ISCO ring in the equatorial plane, the spin
axis and the line of sight are drawn for reference.  Hovering a point shows its Boyer-Lindquist
coordinates, the image-plane pixel it belongs to and its Mino time / coordinate time.

Usage (from project root):
  python python/caustic_3d_plotly.py [fits_file] [output_html] [--nmax N] [--rmax R] [--max-points M]

Defaults: fits_file = dat/caustic_3d.fits, output_html = dat/caustic_3d.html
"""

import argparse
import numpy as np
import astropy.io.fits as pyfits
import plotly.graph_objects as go

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fits_file", nargs="?", default="dat/caustic_3d.fits")
ap.add_argument("out_html", nargs="?", default="dat/caustic_3d.html")
ap.add_argument("--nmax", type=int, default=4, help="highest crossing index to plot (default 4)")
ap.add_argument("--rmax", type=float, default=None, help="only plot points with r < rmax (default: 2 x image-plane extent)")
ap.add_argument("--max-points", type=int, default=200000, help="subsample each order to at most this many points")
args = ap.parse_args()

with pyfits.open(args.fits_file) as f:
    hdr = f[0].header
    spin = hdr["SPIN"]; incl = hdr["INCL"]; phi0 = hdr.get("PHI0", 0.0)
    r_hor = hdr["HORIZON"]; r_isco = hdr["ISCO"]
    x0 = hdr["X0"]; xmax = hdr["XMAX"]; y0 = hdr["Y0"]; ymax = hdr["YMAX"]
    t = f["CAUSTICS"].data

i = np.radians(incl)
n_hat = np.array([np.sin(i) * np.cos(phi0), np.sin(i) * np.sin(phi0), np.cos(i)])
e_x = np.array([-np.sin(phi0), np.cos(phi0), 0.0])
e_y = np.array([-np.cos(i) * np.cos(phi0), -np.cos(i) * np.sin(phi0), np.sin(i)])
R = np.vstack([e_x, e_y, n_hat])          # rows: observer-frame basis

X = np.vstack([t["X"], t["Y"], t["Z"]]).T
P = X @ R.T                               # columns: X', Y', Z'
N = np.array(t["N"]); r = np.array(t["R"])
rmax = args.rmax if args.rmax is not None else 2 * max(abs(x0), abs(xmax), abs(y0), abs(ymax))

# one sequential ramp (viridis) for the crossing index, fixed per order
ramp = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725", "#f0a000", "#c44e52", "#8172b3"]

fig = go.Figure()
for k in range(1, args.nmax + 1):
    m = (N == k) & (r < rmax)
    idx = np.where(m)[0]
    if len(idx) == 0:
        continue
    if len(idx) > args.max_points:
        idx = np.random.default_rng(0).choice(idx, args.max_points, replace=False)
    hover = ("r = %{customdata[0]:.3f}<br>theta = %{customdata[1]:.3f}<br>phi = %{customdata[2]:.3f}"
             "<br>pixel (%{customdata[3]}, %{customdata[4]}) x = %{customdata[5]:.3f} y = %{customdata[6]:.3f}"
             "<br>tau = %{customdata[7]:.4f}  t = %{customdata[8]:.2f}<br>eq. crossings %{customdata[9]}")
    cd = np.vstack([t["R"][idx], t["THETA"][idx], t["PHI"][idx], t["IX"][idx], t["IY"][idx],
                    t["XIMG"][idx], t["YIMG"][idx], t["TAU"][idx], t["T"][idx], t["EQCROSS"][idx]]).T
    fig.add_trace(go.Scatter3d(x=P[idx, 0], y=P[idx, 1], z=P[idx, 2], mode="markers",
                               marker=dict(size=1.8 if k == 1 else 2.2, color=ramp[(k - 1) % len(ramp)], opacity=0.7),
                               name=f"N = {k} ({len(idx)} points)", customdata=cd, hovertemplate=hover))

# event horizon
u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:24j]
S = np.stack([r_hor * np.cos(u) * np.sin(v), r_hor * np.sin(u) * np.sin(v), r_hor * np.cos(v)], axis=-1)
Sp = S @ R.T
fig.add_trace(go.Surface(x=Sp[..., 0], y=Sp[..., 1], z=Sp[..., 2], showscale=False, opacity=0.35,
                         colorscale=[[0, "#888888"], [1, "#888888"]], name="horizon", hoverinfo="skip"))
# ISCO ring (equatorial plane)
ph = np.linspace(0, 2 * np.pi, 200)
ring = np.stack([np.sqrt(r_isco**2 + spin**2) * np.cos(ph), np.sqrt(r_isco**2 + spin**2) * np.sin(ph), 0 * ph], axis=-1) @ R.T
fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines", line=dict(color="#555555", width=2),
                           name="ISCO", hoverinfo="skip"))
# spin axis and line of sight
L = 0.6 * rmax
ax_pts = np.array([[0, 0, -L], [0, 0, L]]) @ R.T
fig.add_trace(go.Scatter3d(x=ax_pts[:, 0], y=ax_pts[:, 1], z=ax_pts[:, 2], mode="lines", line=dict(color="#222222", width=3),
                           name="spin axis", hoverinfo="skip"))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, L], mode="lines", line=dict(color="#999999", width=2, dash="dash"),
                           name="line of sight", hoverinfo="skip"))

fig.update_layout(
    title=f"Kerr caustics: a = {spin}, observer inclination {incl:g}°  (observer frame; Z' towards observer)",
    scene=dict(xaxis_title="X' (image x) [rg]", yaxis_title="Y' (image y) [rg]", zaxis_title="Z' (towards observer) [rg]",
               aspectmode="cube",
               xaxis=dict(range=[-rmax, rmax]), yaxis=dict(range=[-rmax, rmax]), zaxis=dict(range=[-rmax, rmax])),
    legend=dict(itemsizing="constant"),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig.write_html(args.out_html, include_plotlyjs="cdn")
print("Wrote", args.out_html)
