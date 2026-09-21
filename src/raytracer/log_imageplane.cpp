/*
 * log_imageplane.cpp
 */

#include "log_imageplane.h"
#include <stdexcept>
#include <limits>

template <typename T>
int LogImagePlane<T>::total_bins(T lo, T hi, int Nlog, int Nlin)
{
    if (!(lo > T(0)))
        throw std::invalid_argument("LogImagePlane: inner bound (x_min/y_min) must be > 0");
    if (!(hi > lo))
        throw std::invalid_argument("LogImagePlane: outer bound (x_max/y_max) must exceed the inner bound");
    if (Nlog < 1)
        throw std::invalid_argument("LogImagePlane: need at least 1 log-spaced bin per side (Nx/Ny)");
    if (Nlin < 1)
        throw std::invalid_argument("LogImagePlane: need at least 1 linear-zone bin (Nlinx/Nliny)");

    return 2 * Nlog + Nlin;
}

template <typename T>
void LogImagePlane<T>::build_axis(T lo, T hi, int Nlog, int Nlin, std::vector<T>& centers, std::vector<T>& widths) const
{
    const int Ntotal = 2 * Nlog + Nlin;
    centers.resize(Ntotal);
    widths.resize(Ntotal);

    // Positive log zone: Nlog bins spanning [lo, hi], edges e_k = lo*(hi/lo)^(k/Nlog), geometric-mean
    // centers -- computed once here, then reused (mirrored) for the negative zone below.
    std::vector<T> pos_c(Nlog), pos_w(Nlog);
    const T ratio = pow(hi / lo, T(1) / Nlog);
    T e_prev = lo;
    for (int k = 0; k < Nlog; k++)
    {
        const T e_next = lo * pow(ratio, T(k + 1));
        pos_c[k] = sqrt(e_prev * e_next);
        pos_w[k] = e_next - e_prev;
        e_prev = e_next;
    }

    // Negative log zone: exact mirror of the positive zone, written in ascending order (-hi .. -lo).
    for (int k = 0; k < Nlog; k++)
    {
        centers[k] = -pos_c[Nlog - 1 - k];
        widths[k]  =  pos_w[Nlog - 1 - k];
    }

    // Linear zone: Nlin uniform bins spanning [-lo, +lo] -- closes the gap through the origin.
    const T dlin = (T(2) * lo) / Nlin;
    for (int k = 0; k < Nlin; k++)
    {
        centers[Nlog + k] = -lo + (T(k) + T(0.5)) * dlin;
        widths[Nlog + k]  = dlin;
    }

    // Positive log zone, ascending order (lo .. hi).
    for (int k = 0; k < Nlog; k++)
    {
        centers[Nlog + Nlin + k] = pos_c[k];
        widths[Nlog + Nlin + k]  = pos_w[k];
    }
}

template <typename T>
LogImagePlane<T>::LogImagePlane(T dist, T inc,
                                 T x_min, T x_max, int Nx, int Nlinx,
                                 T y_min, T y_max, int Ny, int Nliny,
                                 T spin, T phi, T precision)
    : ImagePlane<T>(dist, inc,
                     T(1), static_cast<T>(total_bins(x_min, x_max, Nx, Nlinx)), T(1),
                     T(1), static_cast<T>(total_bins(y_min, y_max, Ny, Nliny)), T(1),
                     spin, phi, precision),
      m_x_min(x_min), m_x_max(x_max), m_y_min(y_min), m_y_max(y_max),
      m_Nx(Nx), m_Nlinx(Nlinx), m_Ny(Ny), m_Nliny(Nliny),
      m_Nx_total(2 * Nx + Nlinx), m_Ny_total(2 * Ny + Nliny)
{
    // ImagePlane's constructor above has already allocated Raytracer::rays[] at the right size (its own
    // ray-count formula, applied to the dummy x0=y0=1, dx=dy=1, xmax=Nx_total, ymax=Ny_total args, gives
    // exactly Nx_total*Ny_total) and filled it with a throwaway linear-grid fill using its own init_ray.
    // x0=y0=1 (not 0) so that throwaway fill never touches x=y=0 either. Overwrite it now with the real
    // logarithmic grid.
    build_axis(x_min, x_max, Nx, Nlinx, m_xc, m_dx);
    build_axis(y_min, y_max, Ny, Nliny, m_yc, m_dy);

    for (int i = 0; i < m_Nx_total; i++)
    {
        for (int j = 0; j < m_Ny_total; j++)
        {
            const int ix = i * m_Ny_total + j;
            init_ray(Raytracer<T>::rays[ix], m_xc[i], m_yc[j]);
        }
    }
}

template <typename T>
void LogImagePlane<T>::init_ray(Ray<T>& ray, T x, T y, T D, T incl, T phi0) const
{
    // Guard ImagePlane::init_ray's 0/0 (beta = asin(y/b), b = sqrt(x*x+y*y)) at exactly x=y=0 --
    // unreachable from ImagePlane's own linear grid, but reachable from this class's linear zone, so
    // imageplane.cpp has no reason to guard it. Nudge off the origin by an amount physically negligible
    // next to any x_min > 0 used to build the grid, rather than touching that file.
    if (x == T(0) && y == T(0))
        x = std::numeric_limits<T>::epsilon();

    ImagePlane<T>::init_ray(ray, x, y, D, incl, phi0);
}

template <typename T>
void LogImagePlane<T>::init_ray(Ray<T>& ray, T x, T y) const
{
    init_ray(ray, x, y, this->get_dist(), this->get_incl() * M_PI / 180, this->get_phi0());
}

template class LogImagePlane<double>;
template class LogImagePlane<float>;
