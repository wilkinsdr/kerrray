/*
 * adaptive_plane.h
 *
 *  Adaptive refinement of the observer's image plane for the caustic applications (caustic_3d, caustic_ent).
 *
 *  Level 0 is the regular (Nx+1) x (Ny+1) grid of the ImagePlane (point index = ix * (Ny+1) + iy, the ray
 *  ordering of ImagePlane).  A cell is a square of four grid points; a leaf cell flagged by the application's
 *  criterion (e.g. its corners differ in the number of caustic crossings, or the sign of the disc-crossing
 *  Jacobian changes along one of its edges) is split into four children by adding the edge mid-points and the
 *  centre, up to `levels` times, so that the pixel spacing near the critical curve / the caustic contours is
 *  dx / 2^levels while the rest of the plane stays coarse.  Points are keyed by their integer coordinates on
 *  the finest grid, so a point shared by neighbouring cells is traced once.  The application traces the new
 *  points after each `refine()` call and stores its per-point data in vectors indexed by point index.
 *
 *  The set of leaf cells (never split) is a conforming-enough mesh for contour extraction with
 *  zero_contours_cells() (disc_caustic_contour.h): hanging nodes only occur on the edge between a leaf cell
 *  and a refined neighbour, and the refinement criteria used here flag every cell on which the contour
 *  changes sign, so both sides of a sign-changing edge are refined to the same level.
 */

#ifndef ADAPTIVE_PLANE_H_
#define ADAPTIVE_PLANE_H_

#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>

struct PlanePoint
{
    long long i, j;   // integer coordinates on the finest grid (spacing dx / 2^levels)
    int level;        // level at which the point was created (0 = regular grid)
    double x, y;      // image-plane position (rg)
};

struct PlaneCell
{
    int level;
    long long i0, j0, size;   // lower-left corner and side in finest-grid units
    int corner[4];            // point indices: bottom-left, bottom-right, top-right, top-left
    bool split;
};

class AdaptivePlane
{
public:
    AdaptivePlane(double x0, double y0, double dx, double dy, int Nx, int Ny, int levels)
        : m_x0(x0), m_y0(y0), m_dx(dx), m_dy(dy), m_Nx(Nx), m_Ny(Ny), m_levels(levels), m_scale(1LL << levels)
    {
        // regular grid points in ImagePlane order (ix * (Ny+1) + iy)
        for (int ix = 0; ix <= Nx; ix++)
            for (int iy = 0; iy <= Ny; iy++)
                add_point(ix * m_scale, iy * m_scale, 0);
        for (int ix = 0; ix < Nx; ix++)
            for (int iy = 0; iy < Ny; iy++)
            {
                PlaneCell c;
                c.level = 0; c.i0 = ix * m_scale; c.j0 = iy * m_scale; c.size = m_scale; c.split = false;
                c.corner[0] = point_index(ix, iy);     c.corner[1] = point_index(ix + 1, iy);
                c.corner[2] = point_index(ix + 1, iy + 1); c.corner[3] = point_index(ix, iy + 1);
                m_cells.push_back(c);
            }
    }

    int npoints() const { return static_cast<int>(m_points.size()); }
    int nbase() const { return (m_Nx + 1) * (m_Ny + 1); }
    int levels() const { return m_levels; }
    const PlanePoint& point(int k) const { return m_points[k]; }
    const std::vector<PlanePoint>& points() const { return m_points; }
    int point_index(int ix, int iy) const { return ix * (m_Ny + 1) + iy; }   // regular grid
    double fine_dx() const { return m_dx / m_scale; }
    double fine_dy() const { return m_dy / m_scale; }

    // Split every unsplit cell of the given level for which flag(cell) is true (cells of lower level are not
    // revisited).  Returns the indices of the points created; the caller traces them and extends its
    // per-point arrays to npoints().
    std::vector<int> refine(int level, const std::function<bool(const PlaneCell&)>& flag)
    {
        std::vector<int> created;
        if (level >= m_levels) return created;
        const size_t ncells = m_cells.size();
        for (size_t c = 0; c < ncells; c++)
        {
            if (m_cells[c].level != level || m_cells[c].split) continue;
            if (!flag(m_cells[c])) continue;
            m_cells[c].split = true;
            const long long i0 = m_cells[c].i0, j0 = m_cells[c].j0, h = m_cells[c].size / 2;
            const int p[3][3] = {
                { get_or_add(i0,       j0,       level + 1, created), get_or_add(i0 + h, j0,       level + 1, created), get_or_add(i0 + 2*h, j0,       level + 1, created) },
                { get_or_add(i0,       j0 + h,   level + 1, created), get_or_add(i0 + h, j0 + h,   level + 1, created), get_or_add(i0 + 2*h, j0 + h,   level + 1, created) },
                { get_or_add(i0,       j0 + 2*h, level + 1, created), get_or_add(i0 + h, j0 + 2*h, level + 1, created), get_or_add(i0 + 2*h, j0 + 2*h, level + 1, created) } };
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                {
                    PlaneCell k;
                    k.level = level + 1; k.i0 = i0 + b*h; k.j0 = j0 + a*h; k.size = h; k.split = false;
                    k.corner[0] = p[a][b]; k.corner[1] = p[a][b+1]; k.corner[2] = p[a+1][b+1]; k.corner[3] = p[a+1][b];
                    m_cells.push_back(k);
                }
        }
        return created;
    }

    std::vector<PlaneCell> leaf_cells() const
    {
        std::vector<PlaneCell> out;
        for (auto& c : m_cells) if (!c.split) out.push_back(c);
        return out;
    }
    std::vector<PlaneCell> cells_at_level(int level) const
    {
        std::vector<PlaneCell> out;
        for (auto& c : m_cells) if (c.level == level && !c.split) out.push_back(c);
        return out;
    }
    // point index at finest-grid coordinates, or -1
    int find(long long i, long long j) const
    {
        auto it = m_index.find(key(i, j));
        return (it == m_index.end()) ? -1 : it->second;
    }

private:
    double m_x0, m_y0, m_dx, m_dy;
    int m_Nx, m_Ny, m_levels;
    long long m_scale;
    std::vector<PlanePoint> m_points;
    std::vector<PlaneCell> m_cells;
    std::unordered_map<long long, int> m_index;

    long long key(long long i, long long j) const { return i * (1LL << 32) + j; }
    int add_point(long long i, long long j, int level)
    {
        PlanePoint p;
        p.i = i; p.j = j; p.level = level;
        p.x = m_x0 + m_dx * static_cast<double>(i) / m_scale;
        p.y = m_y0 + m_dy * static_cast<double>(j) / m_scale;
        m_points.push_back(p);
        const int idx = static_cast<int>(m_points.size()) - 1;
        m_index[key(i, j)] = idx;
        return idx;
    }
    int get_or_add(long long i, long long j, int level, std::vector<int>& created)
    {
        const int k = find(i, j);
        if (k >= 0) return k;
        const int idx = add_point(i, j, level);
        created.push_back(idx);
        return idx;
    }
};

#endif /* ADAPTIVE_PLANE_H_ */
