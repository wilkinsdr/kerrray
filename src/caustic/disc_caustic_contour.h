/*
 * disc_caustic_contour.h
 *
 *  Zero-contour extraction of a scalar field sampled on the points of a cell mesh (the regular image-plane
 *  grid, or the leaf cells of an adaptively refined plane, adaptive_plane.h), where the field may be
 *  undefined (NaN) at some points.  Used by caustic_ent to trace the caustic / disc-plane intersection as the
 *  zero contour of the ray-family Jacobian J_n(x, y) evaluated at the n-th equatorial crossing of every ray.
 *
 *  Marching squares: every cell edge whose two end values have opposite signs (an exact zero counts as
 *  positive) carries one contour vertex, positioned by linear interpolation of the field; the vertices of a
 *  cell are joined into segments by the standard 16-case table, a saddle (vertices on all four edges) being
 *  resolved by the sign of the cell-centre average.  Cells with an undefined corner are skipped.  Segments are
 *  then linked into ordered polylines (open, or closed when the walk returns to its start).
 *
 *  Vertices are keyed by the pair of point indices of their edge, so an edge shared by two cells carries one
 *  vertex and the contour is continuous across it.  On a refined mesh a hanging node (an edge of a coarse cell
 *  that is split in the neighbouring cells) breaks the link only if the coarse edge changes sign, which the
 *  refinement criteria of the applications avoid by refining both sides of every sign-changing edge.
 */

#ifndef DISC_CAUSTIC_CONTOUR_H_
#define DISC_CAUSTIC_CONTOUR_H_

#include <cmath>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>

#include "adaptive_plane.h"

// a contour vertex on the edge between mesh points p0 and p1; f = interpolation fraction from p0 to p1
struct ContourVertex
{
    int p0, p1;
    double f;
    double x, y;           // interpolated position
};

struct ContourCurve
{
    std::vector<ContourVertex> vertices;
    bool closed;
};

// Extract the zero contours of the field values[point] on the given cells (corners ordered bottom-left,
// bottom-right, top-right, top-left) as ordered polylines.
static inline std::vector<ContourCurve>
zero_contours_cells(const std::vector<PlaneCell>& cells, const std::vector<double>& values,
                    const std::vector<PlanePoint>& pts)
{
    auto defined = [&](int p) { return std::isfinite(values[p]); };
    auto positive = [&](int p) { return values[p] >= 0; };

    // --- one vertex per edge with a sign change ---
    std::map<std::pair<int, int>, int> edge_vertex;
    std::vector<ContourVertex> verts;
    auto make_vertex = [&](int p, int q) -> int
    {
        const std::pair<int, int> key(std::min(p, q), std::max(p, q));
        auto it = edge_vertex.find(key);
        if (it != edge_vertex.end()) return it->second;
        if (!(defined(p) && defined(q)) || positive(p) == positive(q)) { edge_vertex[key] = -1; return -1; }
        const double a = values[key.first], b = values[key.second];
        ContourVertex v;
        v.p0 = key.first; v.p1 = key.second;
        v.f = (a == b) ? 0.5 : a / (a - b);
        v.f = std::min(1.0, std::max(0.0, v.f));
        v.x = pts[v.p0].x + v.f * (pts[v.p1].x - pts[v.p0].x);
        v.y = pts[v.p0].y + v.f * (pts[v.p1].y - pts[v.p0].y);
        verts.push_back(v);
        edge_vertex[key] = static_cast<int>(verts.size()) - 1;
        return edge_vertex[key];
    };

    // --- segments per cell (marching squares) ---
    std::vector<std::pair<int, int>> segs;
    for (const PlaneCell& c : cells)
    {
        const int* k = c.corner;
        if (!(defined(k[0]) && defined(k[1]) && defined(k[2]) && defined(k[3]))) continue;
        // edges: 0 bottom (k0-k1), 1 right (k1-k2), 2 top (k3-k2), 3 left (k0-k3)
        const int e[4] = { make_vertex(k[0], k[1]), make_vertex(k[1], k[2]), make_vertex(k[3], k[2]), make_vertex(k[0], k[3]) };
        int have[4], nh = 0;
        for (int m = 0; m < 4; m++) if (e[m] >= 0) have[nh++] = m;
        if (nh == 2)
            segs.push_back({e[have[0]], e[have[1]]});
        else if (nh == 4)
        {
            // saddle: pair the edges according to the sign of the cell-centre average relative to the
            // bottom-left corner
            const double centre = 0.25 * (values[k[0]] + values[k[1]] + values[k[2]] + values[k[3]]);
            const bool same = ((centre >= 0) == positive(k[0]));
            if (same) { segs.push_back({e[0], e[1]}); segs.push_back({e[2], e[3]}); }
            else      { segs.push_back({e[0], e[3]}); segs.push_back({e[1], e[2]}); }
        }
    }

    // --- link segments into polylines ---
    const int nv = static_cast<int>(verts.size());
    std::vector<std::vector<int>> adj(nv);
    for (auto& s : segs) { adj[s.first].push_back(s.second); adj[s.second].push_back(s.first); }
    std::vector<bool> used(nv, false);
    std::vector<ContourCurve> curves;

    auto walk = [&](int start, ContourCurve& c) -> int
    {
        int prev = -1, cur = start, last = start;
        while (cur >= 0 && !used[cur])
        {
            used[cur] = true;
            c.vertices.push_back(verts[cur]);
            last = cur;
            int next = -1;
            for (int n : adj[cur]) if (n != prev && !used[n]) { next = n; break; }
            prev = cur; cur = next;
        }
        return last;
    };

    // open curves first (start from vertices of degree 1), then closed loops
    for (int pass = 0; pass < 2; pass++)
        for (int v = 0; v < nv; v++)
        {
            if (used[v] || adj[v].empty()) continue;
            if (pass == 0 && adj[v].size() != 1) continue;
            ContourCurve c;
            c.closed = false;
            const int last = walk(v, c);
            if (pass == 1 && c.vertices.size() > 2)
                c.closed = (std::find(adj[last].begin(), adj[last].end(), v) != adj[last].end());
            if (c.vertices.size() >= 2) curves.push_back(c);
        }

    return curves;
}

// Regular grid convenience wrapper: field[iy*nx + ix] at (x0 + ix*dx, y0 + iy*dy).
static inline std::vector<ContourCurve>
zero_contours(const std::vector<double>& field, int nx, int ny, double x0, double y0, double dx, double dy)
{
    std::vector<PlanePoint> pts(nx * ny);
    std::vector<double> values(nx * ny);
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
        {
            PlanePoint& p = pts[iy*nx + ix];
            p.i = ix; p.j = iy; p.level = 0; p.x = x0 + ix*dx; p.y = y0 + iy*dy;
            values[iy*nx + ix] = field[iy*nx + ix];
        }
    std::vector<PlaneCell> cells;
    for (int iy = 0; iy < ny - 1; iy++)
        for (int ix = 0; ix < nx - 1; ix++)
        {
            PlaneCell c;
            c.level = 0; c.i0 = ix; c.j0 = iy; c.size = 1; c.split = false;
            c.corner[0] = iy*nx + ix; c.corner[1] = iy*nx + ix + 1; c.corner[2] = (iy+1)*nx + ix + 1; c.corner[3] = (iy+1)*nx + ix;
            cells.push_back(c);
        }
    return zero_contours_cells(cells, values, pts);
}

#endif /* DISC_CAUSTIC_CONTOUR_H_ */
