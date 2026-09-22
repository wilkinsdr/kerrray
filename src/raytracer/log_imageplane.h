/*
 * log_imageplane.h
 *
 * LogImagePlane: an ImagePlane variant whose (x, y) grid is logarithmically spaced on each axis
 * (mirrored about zero, with a linear zone spanning the origin so the grid has no gap in coverage --
 * "symlog", matplotlib's terminology), rather than ImagePlane's own uniformly-spaced grid. Concentrates
 * resolution near the optical axis (black hole shadow / photon ring) while still covering a wide field of
 * view cheaply. Since pixels are no longer equal-area, each ray carries an explicit area weight
 * (ray_weight()) for use by applications that sum/bin quantities over the grid.
 *
 * Derives from ImagePlane<T> (rather than duplicating its photon-initialisation physics) because
 * ImagePlane::init_ray is public, const, and depends only on (x, y, D, incl, phi0) plus the protected
 * Raytracer::spin -- never on ImagePlane's own private grid state -- so it is already reusable as-is via
 * inheritance. imageplane.h/imageplane.cpp are not modified by this file.
 */

#ifndef LOG_IMAGEPLANE_H_
#define LOG_IMAGEPLANE_H_

#include <vector>
#include "imageplane.h"

template <typename T>
class LogImagePlane : public ImagePlane<T>
{
private:
    T m_x_min, m_x_max, m_y_min, m_y_max;
    int m_Nx, m_Nlinx, m_Ny, m_Nliny;
    int m_Nx_total, m_Ny_total;

    std::vector<T> m_xc, m_dx;   // per-column center coordinate & cell width, size m_Nx_total
    std::vector<T> m_yc, m_dy;   // per-row    center coordinate & cell width, size m_Ny_total

    // Validates (lo>0, hi>lo, Nlog>=1, Nlin>=1) and returns the total point count 2*Nlog+Nlin for one
    // axis -- called directly in the base ImagePlane constructor's argument list, so validation happens
    // before any Raytracer/ImagePlane construction occurs.
    static int total_bins(T lo, T hi, int Nlog, int Nlin);

    // Builds one axis's ascending, gap-free "log - linear - log" bin centers/widths spanning [-hi, hi]:
    // Nlog log-spaced bins each side of [lo, hi], plus Nlin uniform bins spanning [-lo, lo] through zero.
    void build_axis(T lo, T hi, int Nlog, int Nlin, std::vector<T>& centers, std::vector<T>& widths) const;

public:
    // x_min/y_min > 0: inner edge of each axis's log zones. x_max/y_max: outer edge (field-of-view extent).
    // Nx/Ny: number of log-spaced bins per side, per axis. Nlinx/Nliny: number of (uniform) bins in the
    // linear zone spanning -x_min..x_min / -y_min..y_min. Total grid is (2*Nx+Nlinx) x (2*Ny+Nliny) rays.
    LogImagePlane(T dist, T inc,
                  T x_min, T x_max, int Nx, int Nlinx,
                  T y_min, T y_max, int Ny, int Nliny,
                  T spin, T phi, T precision = PRECISION);

    // Same photon-initialisation physics as ImagePlane::init_ray (reused via ImagePlane<T>::init_ray),
    // with a guard for the one case ImagePlane's own grid never reaches: exactly x = y = 0, where
    // ImagePlane::init_ray's beta = asin(y/b), b = sqrt(x*x+y*y), is a 0/0. LogImagePlane's linear zone
    // can legitimately land a ray there, so the origin is nudged by a negligible amount instead. override,
    // not just hiding, now that ImagePlane::init_ray is virtual -- this guard must be reachable through a
    // const ImagePlane<T>& reference (e.g. RayTransfer, src/ray_transfer/ray_transfer.h), not just when
    // called on a LogImagePlane directly.
    void init_ray(Ray<T>& ray, T x, T y, T D, T incl, T phi0) const override;
    void init_ray(Ray<T>& ray, T x, T y) const override;

    int get_Nx() const override { return m_Nx_total; }   // total columns, 2*Nx+Nlinx
    int get_Ny() const override { return m_Ny_total; }   // total rows,    2*Ny+Nliny

    inline int get_x_index(int ix) const { return ix / m_Ny_total; }
    inline int get_y_index(int ix) const { return ix % m_Ny_total; }

    inline T ray_x(int ix) const { return m_xc[get_x_index(ix)]; }
    inline T ray_y(int ix) const { return m_yc[get_y_index(ix)]; }

    // Area on the image plane (Rg^2) represented by ray ix's cell (dx_i * dy_j from the log/linear bin
    // edges). Applications that sum/average flux over rays assuming equal-area pixels should instead
    // weight each ray's contribution by this before summing.
    inline T ray_weight(int ix) const { return m_dx[get_x_index(ix)] * m_dy[get_y_index(ix)]; }

    // ImagePlane's pixel-center grid query (see imageplane.h), overridden with this class's own log-spaced
    // grid -- by column/row index directly (0..get_Nx()-1 / 0..get_Ny()-1), unlike ray_x/ray_y/ray_weight
    // above which take a flat ray index. This is what lets RayTransfer (ray_transfer.h) drive either grid
    // type through one const ImagePlane<T>& reference with no separate adapter object.
    T pixel_x(int ix) const override { return m_xc[ix]; }
    T pixel_y(int iy) const override { return m_yc[iy]; }
    T pixel_dx(int ix) const override { return m_dx[ix]; }
    T pixel_dy(int iy) const override { return m_dy[iy]; }
    T pixel_weight(int ix, int iy) const override { return m_dx[ix] * m_dy[iy]; }
};

#endif /* LOG_IMAGEPLANE_H_ */
