//
// caustic_discplane.cpp
//
// Sky-plane critical curve mapping for Kerr spacetime.
//
// Traces rays backward from a 2D image plane to the accretion disc and computes
// the Jacobian J = d(x_disc, y_disc)/d(x_img, y_img) at each pixel via central
// finite differences.  Cartesian disc coordinates
// (x_disc = r*cos(phi), y_disc = r*sin(phi)) are used to avoid the phi = ±π
// branch-cut artifact that appears when differencing phi directly.
//
// Two Jacobian computation modes:
//
//   Bundle mode (bundle_eps_frac > 0, default):
//     For each image-plane pixel, 4 satellite rays are traced at offsets
//     ±eps_x and ±eps_y (where eps = eps_frac * pixel_spacing).  The
//     Jacobian is computed from the satellite disc hit positions.  This gives
//     much more accurate local derivatives than the grid-neighbour method and
//     avoids SENTINEL gaps near critical curves where neighbours land in
//     different image-order shells.
//
//   Grid-neighbour mode (bundle_eps_frac = 0):
//     Classical central finite differences using the two adjacent grid
//     pixels in each direction.  Kept as a fallback / comparison.
//
// Critical curves on the sky are the zero-contours of det(J).  These map to
// caustics on the disc under the lens map.
//
// The rdot_flips count (number of radial turning points during propagation) is
// stored per pixel to label image order: 0 = direct image, 1 = first photon
// ring, 2 = second ring, etc.
//
// FITS output extensions:
//   DET_J      -- det(J) in image-plane coordinates; NaN where undefined
//   SIGN_J     -- sign of det(J): +1/-1; 0 where undefined
//   ORDER      -- rdot_flips (image order); -1 where no disc hit
//   HIT        -- 1 if ray hit disc, 0 otherwise
//   RADIUS     -- disc radius (gravitational radii) of the disc hit
//   PHI        -- disc azimuthal angle (radians, range [-pi,pi])
//   X_DISC     -- disc Cartesian x coordinate (rg) = r*cos(phi)
//   Y_DISC     -- disc Cartesian y coordinate (rg) = r*sin(phi)
//   REDSHIFT   -- photon energy ratio E_obs/E_emit (includes gravitational + Doppler)
//
// NOTE on array dimensions:
//   The user specifies Nx, Ny grid steps.  ImagePlane creates (Nx+1)*(Ny+1) rays
//   (fencepost convention).  All output arrays therefore have dimensions
//   img_Nx = Nx+1, img_Ny = Ny+1.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <limits>

using namespace std;

#include "../raytracer/imageplane.h"
#include "../raytracer/imageplane_bundles.h"
#include "../raytracer/ray_destination.h"

#include "../include/fits_output.h"
#include "../include/array.h"
#include "../include/par_file.h"
#include "../include/par_args.h"
#include "../include/kerr.h"

static const double NaN = std::numeric_limits<double>::quiet_NaN();


int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile")
                          ? par_args.get_string_parameter("--parfile")
                          : "../par/caustic_discplane.par";

    ParameterFile par_file(par_filename.c_str());
    string out_filename = par_args.key_exists("--outfile")
                          ? par_args.get_parameter<string>("--outfile")
                          : par_file.get_parameter<string>("outfile");

    double dist  = par_file.get_parameter<double>("dist");
    double incl  = par_args.key_exists("--incl")
                   ? par_args.get_parameter<double>("--incl")
                   : par_file.get_parameter<double>("incl");
    double plane_phi0 = par_file.get_parameter<double>("plane_phi0", 0);
    double spin  = par_args.key_exists("--spin")
                   ? par_args.get_parameter<double>("--spin")
                   : par_file.get_parameter<double>("spin");
    double r_disc = par_file.get_parameter<double>("r_disc");

    double x0   = par_file.get_parameter<double>("x0",   -1 * r_disc);
    double xmax = par_file.get_parameter<double>("xmax",  r_disc);
    int    Nx   = par_file.get_parameter<int>("Nx");
    double y0   = par_file.get_parameter<double>("y0",   x0);
    double ymax = par_file.get_parameter<double>("ymax", xmax);
    int    Ny   = par_file.get_parameter<int>("Ny", Nx);

    string integrator_str = par_file.get_parameter<string>("integrator", "rk45");
    double rk45_tol = par_file.get_parameter<double>("rk45_tol", 1e-8);
    double precision = par_file.get_parameter<double>("precision", PRECISION);
    int show_progress = par_args.key_exists("--show_progress")
                        ? par_args.get_parameter<int>("--show_progress")
                        : par_file.get_parameter<int>("show_progress", 1);

    // Bundle Jacobian: eps_frac > 0 enables the bundle mode.
    // Set to 0 to fall back to grid-neighbour finite differences.
    double bundle_eps_frac = par_file.get_parameter<double>("bundle_eps_frac", 0.01);

    Integrator integrator;
    if (integrator_str == "euler")
    {
        cerr << "Warning: Euler integrator does not support RayDestination; using RK45" << endl;
        integrator = Integrator::RK45;
    }
    else if (integrator_str == "rk4")
        integrator = Integrator::RK4;
    else
        integrator = Integrator::RK45;

    // Step sizes for the ray grid.
    // ImagePlane internally computes Nx_internal = (xmax-x0)/dx + 1 = Nx + 1 (fencepost).
    double dx = (xmax - x0) / Nx;
    double dy = (ymax - y0) / Ny;

    // Actual output array dimensions (one extra point per axis due to fencepost)
    int img_Nx = Nx + 1;
    int img_Ny = Ny + 1;

    double r_isco = kerr_isco<double>(spin, +1);
    cout << "ISCO at r = " << r_isco << endl;
    cout << "Image plane: " << img_Nx << " x " << img_Ny << " = " << img_Nx*img_Ny << " rays" << endl;

    bool use_bundles = (bundle_eps_frac > 0.0);
    if (use_bundles)
        cout << "Bundle Jacobian mode: eps_frac=" << bundle_eps_frac
             << "  (eps_x=" << bundle_eps_frac*dx << " eps_y=" << bundle_eps_frac*dy << " rg)" << endl;
    else
        cout << "Grid-neighbour Jacobian mode" << endl;

    // --- Allocate output arrays (filled by either code path below) ---

    Array2D<int>    hit_map(img_Nx, img_Ny);
    Array2D<double> r_disc_map(img_Nx, img_Ny);
    Array2D<double> phi_map(img_Nx, img_Ny);
    Array2D<double> x_disc_map(img_Nx, img_Ny);
    Array2D<double> y_disc_map(img_Nx, img_Ny);
    Array2D<int>    order_map(img_Nx, img_Ny);
    Array2D<double> redshift_map(img_Nx, img_Ny);
    const double    SENTINEL = 1e30;
    Array2D<double> det_J_map(img_Nx, img_Ny);
    Array2D<double> sign_J_map(img_Nx, img_Ny);

    hit_map.zero();
    r_disc_map.zero();
    phi_map.zero();
    x_disc_map.zero();
    y_disc_map.zero();
    order_map.zero();
    redshift_map.zero();

    long disc_count = 0;

    // --- Destination: disc annulus [r_isco, r_disc] ---
    DiscWithISCODestination<double> disc(r_isco, r_disc);

    // Helper: extract disc Cartesian coords from accumulated phi + r.
    auto disc_xy = [](double phi_acc, double r, double& xd, double& yd) {
        double phi_s = atan2(sin(phi_acc), cos(phi_acc));
        xd = r * cos(phi_s);
        yd = r * sin(phi_s);
    };

    // Helper: check whether a ray constitutes a valid disc hit.
    auto valid_hit = [&](const Ray<double>& ray) -> bool {
        return ray.steps > 0 &&
               ray.r >= r_isco &&
               ray.r <  r_disc &&
               ray.redshift > 0;
    };

    // --- Image order: max of two independent estimators.
    //
    //   phi_ord = floor(|phi_acc| / 2pi):
    //     Each full orbit adds 2pi to phi_acc.  Near-side direct rays have
    //     phi_acc ≈ 0 → 0; far-side direct rays have phi_acc ≈ pi → 0;
    //     first photon ring has phi_acc ≈ 2pi–3pi → 1; etc.
    //     Captures photon-sphere orbits (nearly constant r) that accumulate
    //     large phi without any radial turning points.
    //
    //   r_ord = rdot_flips / 2:
    //     Each inward+outward excursion contributes 2 flips.  Correctly
    //     classifies conventional ring rays that have radial turning points.
    //
    //   Taking max(phi_ord, r_ord) handles both cases robustly.
    auto disc_order = [](const Ray<double>& ray) -> int {
        int phi_ord = (int)(std::fabs(ray.phi) / (2 * M_PI));
        int r_ord   = ray.rdot_flips / 2;
        return std::max(phi_ord, r_ord);
    };

    // =========================================================
    // CODE PATH A: Bundle Jacobian
    // =========================================================
    if (use_bundles)
    {
        ImagePlaneBundles<double> tracer(dist, incl, x0, xmax, dx,
                                         y0, ymax, dy, spin, plane_phi0,
                                         precision, bundle_eps_frac);
        if (integrator == Integrator::RK45)
            tracer.set_rk45_tol(rk45_tol);

        tracer.redshift_start();
        tracer.run_raytrace(&disc, integrator, 1.1 * dist, show_progress);
        tracer.redshift(&disc, true);   // reverse=true for backward-traced ImagePlane

        // Gather data from centre rays
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                const Ray<double>& ray = tracer.rays[tracer.centre_ray(ix, iy)];

                if (valid_hit(ray))
                {
                    double phi_acc = ray.phi;
                    double r       = ray.r;
                    double phi_s   = atan2(sin(phi_acc), cos(phi_acc));
                    hit_map[ix][iy]      = 1;
                    r_disc_map[ix][iy]   = r;
                    phi_map[ix][iy]      = phi_s;
                    x_disc_map[ix][iy]   = r * cos(phi_s);
                    y_disc_map[ix][iy]   = r * sin(phi_s);
                    order_map[ix][iy]    = disc_order(ray);
                    redshift_map[ix][iy] = ray.redshift;
                    ++disc_count;
                }
                else
                {
                    hit_map[ix][iy]      = 0;
                    r_disc_map[ix][iy]   = 0;
                    phi_map[ix][iy]      = 0;
                    x_disc_map[ix][iy]   = 0;
                    y_disc_map[ix][iy]   = 0;
                    order_map[ix][iy]    = -1;
                    redshift_map[ix][iy] = 0;
                }
            }
        }

        cout << disc_count << " rays hit the disc" << endl;

        // Diagnostic: count ray failure modes from centre rays
        long cnt_horizon = 0, cnt_rlim = 0, cnt_steplim = 0,
             cnt_other = 0, cnt_outofrange = 0;
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                const Ray<double>& r = tracer.rays[tracer.centre_ray(ix, iy)];
                if (r.steps > 0 && (r.r < r_isco || r.r >= r_disc || r.redshift <= 0))
                    ++cnt_outofrange;
                else if (r.steps <= 0 || !(r.status & RAY_STATUS_DEST))
                {
                    if      (r.status & RAY_STATUS_HORIZON)  ++cnt_horizon;
                    else if (r.status & RAY_STATUS_RLIM)     ++cnt_rlim;
                    else if (r.status & RAY_STATUS_STEPLIM)  ++cnt_steplim;
                    else ++cnt_other;
                }
            }
        }
        cout << "  -> horizon=" << cnt_horizon << " rlim=" << cnt_rlim
             << " steplim=" << cnt_steplim << " out-of-range=" << cnt_outofrange
             << " other=" << cnt_other << endl;

        // Jacobian from satellite rays
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                det_J_map[ix][iy]  = NaN;
                sign_J_map[ix][iy] = 0;

                if (!hit_map[ix][iy]) continue;

                const Ray<double>& re = tracer.rays[tracer.east_ray (ix, iy)];
                const Ray<double>& rw = tracer.rays[tracer.west_ray (ix, iy)];
                const Ray<double>& rn = tracer.rays[tracer.north_ray(ix, iy)];
                const Ray<double>& rs = tracer.rays[tracer.south_ray(ix, iy)];

                if (!valid_hit(re) || !valid_hit(rw) ||
                    !valid_hit(rn) || !valid_hit(rs)) continue;

                const Ray<double>& rc = tracer.rays[tracer.centre_ray(ix, iy)];
                const int    ord_flips = rc.rdot_flips;
                const double phi_c     = rc.phi;
                // Continuity check: satellite accumulated phi must be within
                // pi/2 of the centre ray's phi.  A larger jump indicates the
                // satellite crossed a geodesic branch boundary (fold of the
                // lens mapping, or a near-photon-sphere extra loop), which
                // would give a wildly wrong finite-difference Jacobian.
                bool order_match = (re.rdot_flips == ord_flips) &&
                                   (rw.rdot_flips == ord_flips) &&
                                   (rn.rdot_flips == ord_flips) &&
                                   (rs.rdot_flips == ord_flips) &&
                                   (fabs(re.phi - phi_c) < M_PI_2) &&
                                   (fabs(rw.phi - phi_c) < M_PI_2) &&
                                   (fabs(rn.phi - phi_c) < M_PI_2) &&
                                   (fabs(rs.phi - phi_c) < M_PI_2);
                if (!order_match)
                {
                    det_J_map[ix][iy]  = SENTINEL;
                    sign_J_map[ix][iy] = 0;
                    continue;
                }

                double xe, ye, xw, yw, xn, yn, xs, ys;
                disc_xy(re.phi, re.r, xe, ye);
                disc_xy(rw.phi, rw.r, xw, yw);
                disc_xy(rn.phi, rn.r, xn, yn);
                disc_xy(rs.phi, rs.r, xs, ys);

                double dxd_da = (xe - xw) / (2 * tracer.eps_x);
                double dxd_db = (xn - xs) / (2 * tracer.eps_y);
                double dyd_da = (ye - yw) / (2 * tracer.eps_x);
                double dyd_db = (yn - ys) / (2 * tracer.eps_y);

                double det = dxd_da * dyd_db - dxd_db * dyd_da;
                det_J_map[ix][iy]  = det;
                sign_J_map[ix][iy] = (det > 0) ? 1.0 : (det < 0) ? -1.0 : 0.0;
            }
        }
    }

    // =========================================================
    // CODE PATH B: Grid-neighbour Jacobian (original method)
    // =========================================================
    else
    {
        ImagePlane<double> raytrace_source(dist, incl, x0, xmax, dx,
                                            y0, ymax, dy, spin, plane_phi0, precision);
        if (integrator == Integrator::RK45)
            raytrace_source.set_rk45_tol(rk45_tol);

        raytrace_source.redshift_start();
        raytrace_source.run_raytrace(&disc, integrator, 1.1 * dist, show_progress);
        raytrace_source.redshift(&disc, true);

        for (int ray = 0; ray < raytrace_source.get_count(); ray++)
        {
            int ix = raytrace_source.get_x_index(ray);
            int iy = raytrace_source.get_y_index(ray);

            if (valid_hit(raytrace_source.rays[ray]))
            {
                double phi_acc = raytrace_source.rays[ray].phi;
                double r       = raytrace_source.rays[ray].r;
                double phi_s   = atan2(sin(phi_acc), cos(phi_acc));
                hit_map[ix][iy]      = 1;
                r_disc_map[ix][iy]   = r;
                phi_map[ix][iy]      = phi_s;
                x_disc_map[ix][iy]   = r * cos(phi_s);
                y_disc_map[ix][iy]   = r * sin(phi_s);
                order_map[ix][iy]    = disc_order(raytrace_source.rays[ray]);
                redshift_map[ix][iy] = raytrace_source.rays[ray].redshift;
                ++disc_count;
            }
            else
            {
                hit_map[ix][iy]      = 0;
                r_disc_map[ix][iy]   = 0;
                phi_map[ix][iy]      = 0;
                x_disc_map[ix][iy]   = 0;
                y_disc_map[ix][iy]   = 0;
                order_map[ix][iy]    = -1;
                redshift_map[ix][iy] = 0;
            }
        }

        cout << disc_count << " rays hit the disc" << endl;

        long cnt_horizon = 0, cnt_rlim = 0, cnt_steplim = 0,
             cnt_other = 0, cnt_outofrange = 0;
        for (int ray = 0; ray < raytrace_source.get_count(); ray++)
        {
            auto& r = raytrace_source.rays[ray];
            if (r.steps > 0 && (r.r < r_isco || r.r >= r_disc || r.redshift <= 0))
                ++cnt_outofrange;
            else if (r.steps <= 0 || !(r.status & RAY_STATUS_DEST))
            {
                if      (r.status & RAY_STATUS_HORIZON)  ++cnt_horizon;
                else if (r.status & RAY_STATUS_RLIM)     ++cnt_rlim;
                else if (r.status & RAY_STATUS_STEPLIM)  ++cnt_steplim;
                else ++cnt_other;
            }
        }
        cout << "  -> horizon=" << cnt_horizon << " rlim=" << cnt_rlim
             << " steplim=" << cnt_steplim << " out-of-range=" << cnt_outofrange
             << " other=" << cnt_other << endl;

        // Central finite differences on grid neighbours
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                det_J_map[ix][iy]  = NaN;
                sign_J_map[ix][iy] = 0;

                if (!hit_map[ix][iy]) continue;
                if (ix == 0 || ix == img_Nx-1 || iy == 0 || iy == img_Ny-1) continue;

                int  ord     = order_map[ix][iy];
                bool all_hit = hit_map[ix+1][iy] && hit_map[ix-1][iy] &&
                               hit_map[ix][iy+1] && hit_map[ix][iy-1];
                if (!all_hit) continue;

                bool order_match = (order_map[ix+1][iy] == ord) &&
                                   (order_map[ix-1][iy] == ord) &&
                                   (order_map[ix][iy+1] == ord) &&
                                   (order_map[ix][iy-1] == ord);
                if (!order_match)
                {
                    det_J_map[ix][iy]  = SENTINEL;
                    sign_J_map[ix][iy] = 0;
                    continue;
                }

                double dxd_dx = (x_disc_map[ix+1][iy] - x_disc_map[ix-1][iy]) / (2*dx);
                double dxd_dy = (x_disc_map[ix][iy+1] - x_disc_map[ix][iy-1]) / (2*dy);
                double dyd_dx = (y_disc_map[ix+1][iy] - y_disc_map[ix-1][iy]) / (2*dx);
                double dyd_dy = (y_disc_map[ix][iy+1] - y_disc_map[ix][iy-1]) / (2*dy);

                double det = dxd_dx * dyd_dy - dxd_dy * dyd_dx;
                det_J_map[ix][iy]  = det;
                sign_J_map[ix][iy] = (det > 0) ? 1.0 : (det < 0) ? -1.0 : 0.0;
            }
        }
    }

    // Post-processing: suppress alternating-sign pixels at geodesic branch boundaries.
    //
    // Near the far-side fold (phi_disc ≈ ±π), adjacent image-plane pixels can belong
    // to opposite geodesic branches (prograde vs retrograde).  Each bundle computes a
    // valid Jacobian for its own branch, but the signs alternate at pixel scale,
    // creating a noisy checkerboard that obscures the critical curves.
    //
    // A pixel that has more opposite-sign neighbours than same-sign neighbours is
    // "isolated" in its local neighbourhood — it sits on the branch boundary at
    // sub-pixel scale.  Setting such pixels to SENTINEL replaces the checkerboard
    // with a clean gap at the branch boundary.
    //
    // The pass uses a snapshot of sign_J so it is not self-referential.
    {
        Array2D<double> sign_copy(img_Nx, img_Ny);
        for (int ix = 0; ix < img_Nx; ix++)
            for (int iy = 0; iy < img_Ny; iy++)
                sign_copy[ix][iy] = sign_J_map[ix][iy];

        const int dix[4] = {-1, 1,  0, 0};
        const int diy[4] = { 0, 0, -1, 1};

        long suppressed = 0;
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                double s = sign_copy[ix][iy];
                if (s == 0.0) continue;   // already undefined

                int n_same = 0, n_opp = 0;
                for (int d = 0; d < 4; d++)
                {
                    int jx = ix + dix[d], jy = iy + diy[d];
                    if (jx < 0 || jx >= img_Nx || jy < 0 || jy >= img_Ny) continue;
                    double sn = sign_copy[jx][jy];
                    if (sn == 0.0) continue;
                    if (sn * s > 0) ++n_same;
                    else            ++n_opp;
                }
                // Suppress if more neighbours have opposite sign (need ≥2 to avoid
                // suppressing genuine single-neighbour sign changes at critical curves)
                if (n_opp > n_same && n_opp >= 2)
                {
                    det_J_map[ix][iy]  = SENTINEL;
                    sign_J_map[ix][iy] = 0.0;
                    ++suppressed;
                }
            }
        }
        cout << suppressed << " alternating-sign pixels suppressed (branch boundary)" << endl;
    }

    // --- FITS output ---

    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Kerr spacetime caustic / critical curve mapping (image plane)");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "caustic_discplane");
    fits.write_keyword("DIST",   "Distance to image plane (rg)", dist);
    fits.write_keyword("INCL",   "Inclination (degrees)", incl);
    fits.write_keyword("SPIN",   "Black hole spin parameter a/M", spin);
    fits.write_keyword("ISCO",   "Innermost stable circular orbit (rg)", r_isco);
    fits.write_keyword("RDISC",  "Outer disc radius (rg)", r_disc);
    fits.write_keyword("NRAYS",  "Total number of rays", img_Nx * img_Ny);
    fits.write_keyword("DISC_N", "Rays that hit the disc", disc_count);
    fits.write_keyword("EPSFRAC","Bundle satellite offset fraction (0=grid-neighbour)", bundle_eps_frac);

    // Helper lambda to write common axis keywords
    auto write_axis_keywords = [&](const char* axis1_label)
    {
        fits.write_keyword("AXIS1", "Quantity along X axis", axis1_label);
        fits.write_keyword("AXIS2", "Quantity along Y axis", "Image plane Y (rg)");
        fits.write_keyword("X0",   "Start of X axis (rg)", x0);
        fits.write_keyword("XMAX", "End of X axis (rg)", xmax);
        fits.write_keyword("DX",   "X step (rg)", dx);
        fits.write_keyword("NX",   "Number of pixels in X", img_Nx);
        fits.write_keyword("Y0",   "Start of Y axis (rg)", y0);
        fits.write_keyword("YMAX", "End of Y axis (rg)", ymax);
        fits.write_keyword("DY",   "Y step (rg)", dy);
        fits.write_keyword("NY",   "Number of pixels in Y", img_Ny);
    };

    // DET_J: Jacobian determinant.
    fits.write_image(det_J_map, img_Nx, img_Ny, false);
    fits.set_ext_name("DET_J");
    fits.write_comment("Jacobian determinant det(d(x_disc,y_disc)/d(x,y)); zero-crossings = critical curves");
    write_axis_keywords("Image plane X (rg)");
    fits.write_keyword("SENTINL", "Value used at image-order boundaries (also critical curves)", SENTINEL);

    // SIGN_J: sign of det(J).
    fits.write_image(sign_J_map, img_Nx, img_Ny, false);
    fits.set_ext_name("SIGN_J");
    fits.write_comment("Sign of Jacobian determinant (+1/-1/0); parity flips at critical curves");
    write_axis_keywords("Image plane X (rg)");

    // ORDER: image order (rdot_flips count).
    Array2D<double> order_out(img_Nx, img_Ny);
    for (int ix = 0; ix < img_Nx; ix++)
        for (int iy = 0; iy < img_Ny; iy++)
            order_out[ix][iy] = static_cast<double>(order_map[ix][iy]);
    fits.write_image(order_out, img_Nx, img_Ny, false);
    fits.set_ext_name("ORDER");
    fits.write_comment("Image order (rdot_flips): 0=direct, 1=first photon ring, -1=no hit");
    write_axis_keywords("Image plane X (rg)");

    // HIT: disc-hit mask.
    Array2D<double> hit_out(img_Nx, img_Ny);
    for (int ix = 0; ix < img_Nx; ix++)
        for (int iy = 0; iy < img_Ny; iy++)
            hit_out[ix][iy] = static_cast<double>(hit_map[ix][iy]);
    fits.write_image(hit_out, img_Nx, img_Ny, false);
    fits.set_ext_name("HIT");
    fits.write_comment("1 if ray hit disc, 0 otherwise");
    write_axis_keywords("Image plane X (rg)");

    // RADIUS: disc emission radius.
    fits.write_image(r_disc_map, img_Nx, img_Ny, false);
    fits.set_ext_name("RADIUS");
    fits.write_comment("Disc emission radius (rg); 0 if no disc hit");
    write_axis_keywords("Image plane X (rg)");
    fits.write_keyword("PIXVAL",  "Pixel value quantity", "RADIUS");
    fits.write_keyword("PIXUNIT", "Pixel value unit",     "RG");

    // PHI: disc azimuthal coordinate.
    fits.write_image(phi_map, img_Nx, img_Ny, false);
    fits.set_ext_name("PHI");
    fits.write_comment("Disc azimuthal angle (radians, range [-pi,pi]); 0 if no disc hit");
    write_axis_keywords("Image plane X (rg)");
    fits.write_keyword("PIXVAL",  "Pixel value quantity", "PHI");
    fits.write_keyword("PIXUNIT", "Pixel value unit",     "RAD");

    // X_DISC: disc Cartesian x coordinate.
    fits.write_image(x_disc_map, img_Nx, img_Ny, false);
    fits.set_ext_name("X_DISC");
    fits.write_comment("Disc Cartesian x coordinate (rg) = r*cos(phi); 0 if no disc hit");
    write_axis_keywords("Image plane X (rg)");
    fits.write_keyword("PIXVAL",  "Pixel value quantity", "X_DISC");
    fits.write_keyword("PIXUNIT", "Pixel value unit",     "RG");

    // Y_DISC: disc Cartesian y coordinate.
    fits.write_image(y_disc_map, img_Nx, img_Ny, false);
    fits.set_ext_name("Y_DISC");
    fits.write_comment("Disc Cartesian y coordinate (rg) = r*sin(phi); 0 if no disc hit");
    write_axis_keywords("Image plane X (rg)");
    fits.write_keyword("PIXVAL",  "Pixel value quantity", "Y_DISC");
    fits.write_keyword("PIXUNIT", "Pixel value unit",     "RG");

    // REDSHIFT: photon energy ratio E_obs/E_emit.
    fits.write_image(redshift_map, img_Nx, img_Ny, false);
    fits.set_ext_name("REDSHIFT");
    fits.write_comment("Photon energy ratio E_obs/E_emit (gravitational + Doppler); 0 if no disc hit");
    write_axis_keywords("Image plane X (rg)");
    fits.write_keyword("PIXVAL",  "Pixel value quantity", "REDSHIFT");
    fits.write_keyword("PIXUNIT", "Pixel value unit",     "E_OBS/E_EMIT");

    fits.close();

    cout << "Done. Output: " << out_filename << endl;
    return 0;
}
