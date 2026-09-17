//
// caustic_plane.cpp
//
// Caustic structure of a Kerr black hole mapped onto a flat source plane
// located behind the BH on the sky (the standard gravitational-lensing
// "source plane").
//
// Traces rays backward from the observer image plane through Kerr spacetime.
// Rays are stopped when they cross the source plane defined by
//
//   d(r,θ,φ) = r*(sin(θ)*sin(i)*cos(φ-φ0) + cos(θ)*cos(i)) = -z_s
//
// where (i, φ0) is the observer direction and z_s is the plane distance
// behind the BH.  Only rays reaching the far side trigger this condition;
// near-side background rays and shadow rays are excluded automatically.
//
// Using a flat Cartesian plane (x_s, y_s) rather than the sphere (θ_s, φ_s)
// avoids the φ = ±π branch-cut artefacts that appear in caustic_sourceplane.
// The Jacobian J = d(x_s, y_s)/d(x_img, y_img) has no coordinate singularity,
// so its zero-crossings cleanly identify critical curves in the image plane
// and caustics on the source plane.
//
// Two Jacobian computation modes:
//
//   Bundle mode (bundle_eps_frac > 0, default):
//     For each image-plane pixel, 4 satellite rays are traced at offsets
//     ±eps_x and ±eps_y.  The Jacobian is computed from the satellite
//     source-plane hit positions.  This gives much more accurate local
//     derivatives and avoids SENTINEL gaps near critical curves.
//
//   Grid-neighbour mode (bundle_eps_frac = 0):
//     Classical central finite differences using adjacent grid pixels.
//
// FITS output extensions:
//   DET_J      -- det(J) = (dx_s/dx)(dy_s/dy) - (dx_s/dy)(dy_s/dx);
//                 zero-crossings are critical curves; NaN where undefined.
//   SIGN_J     -- sign of det(J): +1, -1, or 0; 0 at boundaries/undefined.
//   ORDER      -- image order: 0 = direct, 1 = first photon ring, etc.
//                 -1 if the ray did not hit the source plane.
//   HIT_PLANE  -- 1 if the ray reached the source plane; 0 otherwise.
//   X_S        -- East coordinate on source plane (rg); NaN if not hit.
//   Y_S        -- North coordinate on source plane (rg); NaN if not hit.
//   RDOT_FLIPS -- number of radial turning points during propagation.
//   EQUAT_CROSS -- number of equatorial-plane (θ=π/2) crossings.
//

#include <iostream>
#include <iomanip>
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

static const double NaN = std::numeric_limits<double>::quiet_NaN();


int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile")
                          ? par_args.get_string_parameter("--parfile")
                          : "../par/caustic_plane.par";

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

    // Source plane distance behind the BH (positive, gravitational radii).
    double z_s = par_args.key_exists("--z_s")
                 ? par_args.get_parameter<double>("--z_s")
                 : par_file.get_parameter<double>("z_s", dist);

    // Maximum ray propagation radius.  Must be large enough for deflected rays
    // to reach the source plane.  Default: 4 * z_s.
    double r_max = par_args.key_exists("--r_max")
                   ? par_args.get_parameter<double>("--r_max")
                   : par_file.get_parameter<double>("r_max", 4.0 * z_s);

    double x0   = par_file.get_parameter<double>("x0",   -20.0);
    double xmax = par_file.get_parameter<double>("xmax",  20.0);
    int    Nx   = par_file.get_parameter<int>("Nx");
    double y0   = par_file.get_parameter<double>("y0",   x0);
    double ymax = par_file.get_parameter<double>("ymax", xmax);
    int    Ny   = par_file.get_parameter<int>("Ny", Nx);

    string integrator_str = par_file.get_parameter<string>("integrator", "rk45");
    double rk45_tol  = par_file.get_parameter<double>("rk45_tol", 1e-8);
    int    steplim   = par_args.key_exists("--steplim")
                       ? par_args.get_parameter<int>("--steplim")
                       : par_file.get_parameter<int>("steplim", -1);
    double precision = par_file.get_parameter<double>("precision", PRECISION);
    int show_progress = par_args.key_exists("--show_progress")
                        ? par_args.get_parameter<int>("--show_progress")
                        : par_file.get_parameter<int>("show_progress", 1);

    // Bundle Jacobian: eps_frac > 0 enables the bundle mode.
    // Set to 0 to fall back to grid-neighbour finite differences.
    double bundle_eps_frac = par_file.get_parameter<double>("bundle_eps_frac", 0.01);

    Integrator integrator;
    if (integrator_str == "rk4")
        integrator = Integrator::RK4;
    else
    {
        if (integrator_str != "rk45")
            cerr << "Warning: unknown integrator '" << integrator_str << "'; using RK45" << endl;
        integrator = Integrator::RK45;
    }

    // Step sizes for the ray grid (fencepost: Nx steps → Nx+1 pixels).
    double dx = (xmax - x0) / Nx;
    double dy = (ymax - y0) / Ny;

    int img_Nx = Nx + 1;
    int img_Ny = Ny + 1;

    const double incl_rad = incl * M_PI / 180.0;

    cout << "Image plane: " << img_Nx << " x " << img_Ny
         << " = " << img_Nx * img_Ny << " rays" << endl;
    cout << "Source plane z_s = " << z_s << " rg, r_max = " << r_max << " rg" << endl;

    bool use_bundles = (bundle_eps_frac > 0.0);
    if (use_bundles)
        cout << "Bundle Jacobian mode: eps_frac=" << bundle_eps_frac
             << "  (eps_x=" << bundle_eps_frac*dx << " eps_y=" << bundle_eps_frac*dy << " rg)" << endl;
    else
        cout << "Grid-neighbour Jacobian mode" << endl;

    // --- Flat source plane destination ---
    FlatPlaneDestination<double> dest(incl_rad, plane_phi0, z_s);

    // --- Allocate output arrays ---

    Array2D<int>    hit_map(img_Nx, img_Ny);
    Array2D<double> xs_map(img_Nx, img_Ny);
    Array2D<double> ys_map(img_Nx, img_Ny);
    Array2D<int>    order_map(img_Nx, img_Ny);
    Array2D<int>    rdot_map(img_Nx, img_Ny);
    Array2D<int>    equat_map(img_Nx, img_Ny);
    const double    SENTINEL = 1e30;
    Array2D<double> det_J_map(img_Nx, img_Ny);
    Array2D<double> sign_J_map(img_Nx, img_Ny);

    hit_map.zero();
    xs_map.zero();
    ys_map.zero();
    order_map.zero();
    rdot_map.zero();
    equat_map.zero();

    long hit_count      = 0;
    long captured_count = 0;
    long steplim_count  = 0;

    // Image order: max of phi-winding and rdot-turning-point estimators.
    // phi_ord = floor(|phi_acc| / 2pi): handles photon-sphere orbits (no rdot flips).
    // r_ord = rdot_flips / 2: handles conventional ring rays with turning points.
    // Direct far-side rays have phi_acc ≈ pi → phi_ord = 0 (correct).
    auto plane_order = [](const Ray<double>& ray) -> int {
        int phi_ord = (int)(std::fabs(ray.phi) / (2 * M_PI));
        int r_ord   = ray.rdot_flips / 2;
        return std::max(phi_ord, r_ord);
    };

    // Helper: check whether a ray hit the source plane.
    auto valid_hit = [](const Ray<double>& ray) -> bool {
        return (ray.steps > 0) && (ray.status & RAY_STATUS_DEST);
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

        // No redshift for source-plane mapping (geometry only).
        tracer.run_raytrace(&dest, integrator, r_max, show_progress,
                             nullptr, 1, -1, -1, true, steplim);

        // Gather data from centre rays
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                const Ray<double>& rd = tracer.rays[tracer.centre_ray(ix, iy)];

                rdot_map[ix][iy]  = rd.rdot_flips;
                equat_map[ix][iy] = rd.equatorial_crossings;

                bool hit      = valid_hit(rd);
                bool captured = (rd.status & RAY_STATUS_HORIZON);
                bool step_lim = (rd.steps <= 0) || (rd.status & RAY_STATUS_STEPLIM);

                if (step_lim) ++steplim_count;

                if (hit)
                {
                    double x_s, y_s;
                    dest.source_coords(rd.r, rd.theta, rd.phi, x_s, y_s);
                    hit_map[ix][iy]   = 1;
                    xs_map[ix][iy]    = x_s;
                    ys_map[ix][iy]    = y_s;
                    order_map[ix][iy] = plane_order(rd);
                    ++hit_count;
                }
                else
                {
                    hit_map[ix][iy]   = 0;
                    xs_map[ix][iy]    = NaN;
                    ys_map[ix][iy]    = NaN;
                    order_map[ix][iy] = -1;
                    if (captured) ++captured_count;
                }
            }
        }

        cout << hit_count      << " rays hit source plane" << endl;
        cout << captured_count << " rays captured by BH" << endl;
        if (steplim_count > 0)
            cerr << "Warning: " << steplim_count << " rays hit step limit" << endl;

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
                dest.source_coords(re.r, re.theta, re.phi, xe, ye);
                dest.source_coords(rw.r, rw.theta, rw.phi, xw, yw);
                dest.source_coords(rn.r, rn.theta, rn.phi, xn, yn);
                dest.source_coords(rs.r, rs.theta, rs.phi, xs, ys);

                double dxs_da = (xe - xw) / (2 * tracer.eps_x);
                double dxs_db = (xn - xs) / (2 * tracer.eps_y);
                double dys_da = (ye - yw) / (2 * tracer.eps_x);
                double dys_db = (yn - ys) / (2 * tracer.eps_y);

                double det = dxs_da * dys_db - dxs_db * dys_da;
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

        raytrace_source.run_raytrace(&dest, integrator, r_max, show_progress,
                                      nullptr, 1, -1, -1, true, steplim);

        for (int ray = 0; ray < raytrace_source.get_count(); ray++)
        {
            int ix = raytrace_source.get_x_index(ray);
            int iy = raytrace_source.get_y_index(ray);

            const Ray<double>& rd = raytrace_source.rays[ray];

            rdot_map[ix][iy]  = rd.rdot_flips;
            equat_map[ix][iy] = rd.equatorial_crossings;

            bool hit      = valid_hit(rd);
            bool captured = (rd.status & RAY_STATUS_HORIZON);
            bool step_lim = (rd.steps <= 0) || (rd.status & RAY_STATUS_STEPLIM);

            if (step_lim) ++steplim_count;

            if (hit)
            {
                double x_s, y_s;
                dest.source_coords(rd.r, rd.theta, rd.phi, x_s, y_s);
                hit_map[ix][iy]   = 1;
                xs_map[ix][iy]    = x_s;
                ys_map[ix][iy]    = y_s;
                order_map[ix][iy] = plane_order(rd);
                ++hit_count;
            }
            else
            {
                hit_map[ix][iy]   = 0;
                xs_map[ix][iy]    = NaN;
                ys_map[ix][iy]    = NaN;
                order_map[ix][iy] = -1;
                if (captured) ++captured_count;
            }
        }

        cout << hit_count      << " rays hit source plane" << endl;
        cout << captured_count << " rays captured by BH" << endl;
        if (steplim_count > 0)
            cerr << "Warning: " << steplim_count << " rays hit step limit" << endl;

        // Central finite differences on grid neighbours
        for (int ix = 0; ix < img_Nx; ix++)
        {
            for (int iy = 0; iy < img_Ny; iy++)
            {
                det_J_map[ix][iy]  = NaN;
                sign_J_map[ix][iy] = 0;

                if (!hit_map[ix][iy]) continue;
                if (ix == 0 || ix == img_Nx-1 || iy == 0 || iy == img_Ny-1) continue;

                bool all_hit = hit_map[ix+1][iy] && hit_map[ix-1][iy] &&
                               hit_map[ix][iy+1] && hit_map[ix][iy-1];
                if (!all_hit) continue;

                int ord = order_map[ix][iy];
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

                double dxs_dx = (xs_map[ix+1][iy] - xs_map[ix-1][iy]) / (2*dx);
                double dxs_dy = (xs_map[ix][iy+1] - xs_map[ix][iy-1]) / (2*dy);
                double dys_dx = (ys_map[ix+1][iy] - ys_map[ix-1][iy]) / (2*dx);
                double dys_dy = (ys_map[ix][iy+1] - ys_map[ix][iy-1]) / (2*dy);

                double det = dxs_dx * dys_dy - dxs_dy * dys_dx;
                det_J_map[ix][iy]  = det;
                sign_J_map[ix][iy] = (det > 0) ? 1.0 : (det < 0) ? -1.0 : 0.0;
            }
        }
    }

    // --- FITS output ---

    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Kerr BH caustic / critical curve mapping — flat source plane");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software",
                       "caustic_plane");
    fits.write_keyword("DIST",    "Observer distance (rg)",                  dist);
    fits.write_keyword("INCL",    "Observer inclination (degrees)",          incl);
    fits.write_keyword("PHI0",    "Observer azimuth (radians)",              plane_phi0);
    fits.write_keyword("SPIN",    "Black hole spin parameter a/M",           spin);
    fits.write_keyword("ZS",      "Source plane distance behind BH (rg)",   z_s);
    fits.write_keyword("RMAX",    "Maximum ray propagation radius (rg)",     r_max);
    fits.write_keyword("NRAYS",   "Total number of rays",                    img_Nx * img_Ny);
    fits.write_keyword("N_HIT",   "Rays that hit source plane",              hit_count);
    fits.write_keyword("N_CAP",   "Rays captured by BH",                    captured_count);
    fits.write_keyword("N_SLIM",  "Rays that hit step limit",                steplim_count);
    fits.write_keyword("EPSFRAC", "Bundle satellite offset fraction (0=grid-neighbour)", bundle_eps_frac);

    auto write_axis_kw = [&]()
    {
        fits.write_keyword("X0",   "Start of X axis (rg)", x0);
        fits.write_keyword("XMAX", "End of X axis (rg)",   xmax);
        fits.write_keyword("DX",   "X step (rg)",          dx);
        fits.write_keyword("NX",   "Number of pixels in X", img_Nx);
        fits.write_keyword("Y0",   "Start of Y axis (rg)", y0);
        fits.write_keyword("YMAX", "End of Y axis (rg)",   ymax);
        fits.write_keyword("DY",   "Y step (rg)",          dy);
        fits.write_keyword("NY",   "Number of pixels in Y", img_Ny);
    };

    auto int_to_double = [&](Array2D<int>& src, Array2D<double>& dst) {
        for (int ix = 0; ix < img_Nx; ix++)
            for (int iy = 0; iy < img_Ny; iy++)
                dst[ix][iy] = static_cast<double>(src[ix][iy]);
    };

    Array2D<double> order_out(img_Nx, img_Ny);
    Array2D<double> hit_out(img_Nx, img_Ny);
    Array2D<double> rdot_out(img_Nx, img_Ny);
    Array2D<double> equat_out(img_Nx, img_Ny);
    int_to_double(order_map, order_out);
    int_to_double(hit_map,   hit_out);
    int_to_double(rdot_map,  rdot_out);
    int_to_double(equat_map, equat_out);

    fits.write_image(det_J_map, img_Nx, img_Ny, false);
    fits.set_ext_name("DET_J");
    fits.write_comment("det(J) of image-plane->source-plane map; zero-crossings = critical curves");
    fits.write_comment("NaN: ray did not hit plane; SENTINL: image-order boundary");
    fits.write_keyword("SENTINL", "Value at image-order boundaries (also critical curves)", SENTINEL);
    write_axis_kw();

    fits.write_image(sign_J_map, img_Nx, img_Ny, false);
    fits.set_ext_name("SIGN_J");
    fits.write_comment("Sign of det(J): +1 or -1; 0 at order boundaries or undefined");
    write_axis_kw();

    fits.write_image(order_out, img_Nx, img_Ny, false);
    fits.set_ext_name("ORDER");
    fits.write_comment("Image order: 0=direct, 1=first photon ring; -1=did not hit plane");
    write_axis_kw();

    fits.write_image(hit_out, img_Nx, img_Ny, false);
    fits.set_ext_name("HIT_PLANE");
    fits.write_comment("1 = ray reached source plane; 0 = captured or near-side escape");
    write_axis_kw();

    fits.write_image(xs_map, img_Nx, img_Ny, false);
    fits.set_ext_name("X_S");
    fits.write_comment("East coordinate on source plane (rg); NaN if ray did not hit plane");
    write_axis_kw();

    fits.write_image(ys_map, img_Nx, img_Ny, false);
    fits.set_ext_name("Y_S");
    fits.write_comment("North coordinate on source plane (rg); NaN if ray did not hit plane");
    write_axis_kw();

    fits.write_image(rdot_out, img_Nx, img_Ny, false);
    fits.set_ext_name("RDOT_FLIPS");
    fits.write_comment("Radial turning-point count during propagation");
    write_axis_kw();

    fits.write_image(equat_out, img_Nx, img_Ny, false);
    fits.set_ext_name("EQUAT_CROSS");
    fits.write_comment("Equatorial-plane (theta=pi/2) crossing count during propagation");
    write_axis_kw();

    fits.close();
    cout << "Written to " << out_filename << endl;

    return 0;
}
