//
// caustic_sourceplane.cpp
//
// Caustic structure of a Kerr black hole mapped via a source plane (sphere)
// located behind the BH on the sky.
//
// Traces rays backward from the observer image plane through Kerr spacetime.
// Unlike caustic_discplane (which stops rays at the accretion disc), rays
// here propagate freely until they either escape to a large "source sphere"
// at r = r_lim or are captured by the BH.  This captures all image orders
// and all types of caustic structure for sources at any angular position.
//
// Critical curves and caustics are detected by computing the Jacobian
//
//   J = d(theta_s, phi_s) / d(x_img, y_img)
//
// where (theta_s, phi_s) are the Boyer-Lindquist angular coordinates of the
// ray when it reaches r_lim, and (x_img, y_img) are the image-plane sky
// coordinates.  Critical curves in the image plane are where det(J) = 0.
// Caustics are the images of the critical curves on the source sphere.
//
// Two independent flavours of critical curve are captured:
//   1. Fold caustics — det(J) changes sign within a single image-order band.
//   2. Order-boundary critical curves — where adjacent pixels have different
//      image orders (number of half-orbits around the BH).  These are the
//      boundaries of the photon rings and are flagged with a large sentinel.
//
// Image order is defined as floor(|phi_accumulated| / pi), where phi_accumulated
// is the total azimuthal coordinate accumulated during ray propagation.  This
// counts the number of pi-radian half-orbits made around the BH axis.
//
// Rays propagate with theta_max = 0 in run_raytrace(), which disables the
// equatorial-plane stopping condition in all three propagators (Euler/RK4/RK45)
// and lets rays run until r_lim or the BH horizon.
//
// FITS output extensions:
//   DET_J       -- det(J) = (dtheta/dx)(dphi/dy) - (dtheta/dy)(dphi/dx);
//                  zero-crossings are critical curves in the image plane;
//                  NaN where undefined (captured ray or image boundary).
//   SIGN_J      -- sign of det(J): +1 or -1; 0 at boundaries / undefined.
//   ORDER       -- image order floor(|phi_acc|/pi): 0=direct, 1=1st ring, etc.
//                  -1 if the ray was captured by the BH.
//   ESCAPED     -- 1 if the ray escaped to r_lim; 0 otherwise.
//   THETA_S     -- source polar angle (radians) at r_lim; NaN if captured.
//   PHI_S       -- source azimuthal angle (radians, range [-pi,pi]); NaN if captured.
//   RDOT_FLIPS  -- number of radial turning points during propagation.
//                  Order-n images typically have rdot_flips = 2n+1.
//   EQUAT_CROSS -- number of equatorial-plane (theta=pi/2) crossings.
//

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

using namespace std;

#include "../raytracer/imageplane.h"

#include "../include/fits_output.h"
#include "../include/array.h"
#include "../include/par_file.h"
#include "../include/par_args.h"

static const double NaN = std::numeric_limits<double>::quiet_NaN();

// Wraps a phi difference into [-pi, pi] to handle the phi = +-pi branch cut.
static inline double wrap_dphi(double dphi)
{
    while (dphi >  M_PI) dphi -= 2*M_PI;
    while (dphi < -M_PI) dphi += 2*M_PI;
    return dphi;
}


int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile")
                          ? par_args.get_string_parameter("--parfile")
                          : "../par/caustic_sourceplane.par";

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

    // Source sphere radius.  Rays that reach r_lim are considered escaped.
    // Should be >= dist so that rays returning from the BH side are captured.
    // Default: 1.5 * dist.
    double r_lim = par_args.key_exists("--r_lim")
                   ? par_args.get_parameter<double>("--r_lim")
                   : par_file.get_parameter<double>("r_lim", 1.5 * dist);

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

    cout << "Image plane: " << img_Nx << " x " << img_Ny
         << " = " << img_Nx * img_Ny << " rays" << endl;
    cout << "Source sphere r_lim = " << r_lim << " rg" << endl;

    // --- Ray tracing ---
    //
    // theta_max = 0.0 disables the equatorial-plane stopping condition in all
    // propagators (the while-loop condition becomes (thetalim==0) which is
    // always true, so rays run until r_lim or the BH horizon).

    ImagePlane<double> raytrace_source(dist, incl, x0, xmax, dx, y0, ymax, dy,
                                       spin, plane_phi0, precision);
    if (integrator == Integrator::RK45)
        raytrace_source.set_rk45_tol(rk45_tol);

    // No redshift_start() / redshift(): we are mapping geometry only.
    raytrace_source.run_raytrace(integrator, 0.0, r_lim, show_progress, nullptr, 1, -1, -1, true, steplim);

    // Do NOT call range_phi(): we need the raw accumulated phi for image-order
    // calculation and source-phi normalisation below.

    // --- Gather per-pixel data into 2D arrays ---

    Array2D<int>    escaped_map(img_Nx, img_Ny);
    Array2D<double> theta_s_map(img_Nx, img_Ny);
    Array2D<double> phi_s_map(img_Nx, img_Ny);
    Array2D<int>    order_map(img_Nx, img_Ny);
    Array2D<int>    rdot_map(img_Nx, img_Ny);
    Array2D<int>    equat_map(img_Nx, img_Ny);

    escaped_map.zero();
    theta_s_map.zero();
    phi_s_map.zero();
    order_map.zero();
    rdot_map.zero();
    equat_map.zero();

    long escaped_count  = 0;
    long captured_count = 0;
    long steplim_count  = 0;

    for (int ray = 0; ray < raytrace_source.get_count(); ray++)
    {
        int ix = raytrace_source.get_x_index(ray);
        int iy = raytrace_source.get_y_index(ray);

        const Ray<double>& rd = raytrace_source.rays[ray];

        // Always record the turning-point diagnostics regardless of outcome.
        rdot_map[ix][iy]  = rd.rdot_flips;
        equat_map[ix][iy] = rd.equatorial_crossings;

        bool escaped  = (rd.steps > 0) && (rd.status & RAY_STATUS_RLIM);
        bool captured = (rd.status & RAY_STATUS_HORIZON);
        bool steplim  = (rd.steps <= 0) || (rd.status & RAY_STATUS_STEPLIM);

        if (steplim) ++steplim_count;

        if (escaped)
        {
            // phi is the accumulated (not yet normalised) azimuthal coordinate.
            // Use it to compute the image order and (after wrapping) as the
            // normalised source phi.
            double phi_acc = rd.phi;
            double phi_s   = atan2(sin(phi_acc), cos(phi_acc));   // [-pi, pi]

            // Image order for source-sphere traces:
            //   A backward-traced direct-image ray travels from the observer to
            //   the source on the far side of the BH, naturally accumulating ~pi
            //   in azimuthal angle even with zero extra orbits.  Subtracting 1
            //   before the floor accounts for this, so order=0 means the direct
            //   image, order=1 the first photon ring, etc.  The first meaningful
            //   order boundary (at phi_acc = 2*pi) then falls at the actual photon
            //   ring boundary near the shadow, not spuriously in the outer image.
            int phi_order = (int)floor(fabs(phi_acc) / M_PI);
            int order     = (phi_order > 0) ? phi_order - 1 : 0;

            escaped_map[ix][iy]  = 1;
            theta_s_map[ix][iy]  = rd.theta;
            phi_s_map[ix][iy]    = phi_s;
            order_map[ix][iy]    = order;

            ++escaped_count;
        }
        else
        {
            escaped_map[ix][iy]  = 0;
            theta_s_map[ix][iy]  = NaN;
            phi_s_map[ix][iy]    = NaN;
            order_map[ix][iy]    = -1;

            if (captured) ++captured_count;
        }
    }

    cout << escaped_count  << " rays escaped to source sphere" << endl;
    cout << captured_count << " rays captured by BH" << endl;
    if (steplim_count > 0)
        cerr << "Warning: " << steplim_count
             << " rays hit step limit (consider tightening rk45_tol)" << endl;

    // --- Compute Jacobian det(J) via central finite differences ---
    //
    // Source coordinates: (theta_s, phi_s).
    //
    // For pixel (ix, iy):
    //   dtheta_dx = (theta_s[ix+1][iy] - theta_s[ix-1][iy]) / (2*dx)
    //   dtheta_dy = (theta_s[ix][iy+1] - theta_s[ix][iy-1]) / (2*dy)
    //   dphi_dx   = wrap(phi_s[ix+1][iy] - phi_s[ix-1][iy]) / (2*dx)
    //   dphi_dy   = wrap(phi_s[ix][iy+1] - phi_s[ix][iy-1]) / (2*dy)
    //   det(J)    = dtheta_dx * dphi_dy - dtheta_dy * dphi_dx
    //
    // Defined only when all four cardinal neighbours escaped AND all share the
    // same image order.  At image-order boundaries the mapping is discontinuous;
    // these are also critical curves and are marked with a large SENTINEL value.
    //
    // NOTE: det(J) in (theta, phi) has a sin(theta_s) factor relative to the
    // solid-angle Jacobian.  Zero-crossings (caustics) are located correctly;
    // divide by sin(theta_s) to recover the solid-angle magnification ratio.

    const double SENTINEL = 1e30;

    Array2D<double> det_J_map(img_Nx, img_Ny);
    Array2D<double> sign_J_map(img_Nx, img_Ny);

    for (int ix = 0; ix < img_Nx; ix++)
    {
        for (int iy = 0; iy < img_Ny; iy++)
        {
            det_J_map[ix][iy]  = NaN;
            sign_J_map[ix][iy] = 0;

            // Need a valid escaped pixel not at the image-plane boundary.
            if (!escaped_map[ix][iy]) continue;
            if (ix == 0 || ix == img_Nx-1 || iy == 0 || iy == img_Ny-1) continue;

            // All four cardinal neighbours must also have escaped.
            bool all_escaped = escaped_map[ix+1][iy] && escaped_map[ix-1][iy] &&
                               escaped_map[ix][iy+1] && escaped_map[ix][iy-1];
            if (!all_escaped) continue;

            // All neighbours must share the same image order.
            int ord = order_map[ix][iy];
            bool order_match = (order_map[ix+1][iy] == ord) &&
                               (order_map[ix-1][iy] == ord) &&
                               (order_map[ix][iy+1] == ord) &&
                               (order_map[ix][iy-1] == ord);

            if (!order_match)
            {
                // Pixel sits on an image-order boundary = critical curve.
                det_J_map[ix][iy]  = SENTINEL;
                sign_J_map[ix][iy] = 0;
                continue;
            }

            // Central finite differences.
            double dtheta_dx = (theta_s_map[ix+1][iy] - theta_s_map[ix-1][iy]) / (2*dx);
            double dtheta_dy = (theta_s_map[ix][iy+1] - theta_s_map[ix][iy-1]) / (2*dy);
            double dphi_dx   = wrap_dphi(phi_s_map[ix+1][iy] - phi_s_map[ix-1][iy]) / (2*dx);
            double dphi_dy   = wrap_dphi(phi_s_map[ix][iy+1] - phi_s_map[ix][iy-1]) / (2*dy);

            double det = dtheta_dx * dphi_dy - dtheta_dy * dphi_dx;
            det_J_map[ix][iy]  = det;
            sign_J_map[ix][iy] = (det > 0) ? 1.0 : (det < 0) ? -1.0 : 0.0;
        }
    }

    // --- FITS output ---

    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Kerr BH caustic / critical curve mapping — source sphere");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software",
                       "caustic_sourceplane");
    fits.write_keyword("DIST",    "Observer distance (rg)",             dist);
    fits.write_keyword("INCL",    "Observer inclination (degrees)",     incl);
    fits.write_keyword("SPIN",    "Black hole spin parameter a/M",      spin);
    fits.write_keyword("RLIM",    "Source sphere radius (rg)",          r_lim);
    fits.write_keyword("NRAYS",   "Total number of rays",               img_Nx * img_Ny);
    fits.write_keyword("N_ESC",   "Rays that escaped to source sphere", escaped_count);
    fits.write_keyword("N_CAP",   "Rays captured by BH",                captured_count);
    fits.write_keyword("N_SLIM",  "Rays that hit step limit",           steplim_count);

    // Helper lambda to write per-extension axis keywords.
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

    // DET_J — Jacobian determinant in image-plane coordinates.
    fits.write_image(det_J_map, img_Nx, img_Ny, false);
    fits.set_ext_name("DET_J");
    fits.write_comment("det(J) = (dtheta_s/dx)(dphi_s/dy) - (dtheta_s/dy)(dphi_s/dx)");
    fits.write_comment("Zero-crossings are critical curves (source plane caustics)");
    fits.write_comment("NaN: captured ray or image boundary; SENTINL: order boundary");
    fits.write_keyword("SENTINL", "Value at image-order boundaries (also critical curves)", SENTINEL);
    write_axis_kw();

    // SIGN_J — parity of the Jacobian.
    fits.write_image(sign_J_map, img_Nx, img_Ny, false);
    fits.set_ext_name("SIGN_J");
    fits.write_comment("Sign of det(J): +1 or -1; 0 at order boundaries or undefined");
    fits.write_comment("Image parity flips at every critical curve");
    write_axis_kw();

    // ORDER — image order map.
    Array2D<double> order_out(img_Nx, img_Ny);
    for (int ix = 0; ix < img_Nx; ix++)
        for (int iy = 0; iy < img_Ny; iy++)
            order_out[ix][iy] = static_cast<double>(order_map[ix][iy]);
    fits.write_image(order_out, img_Nx, img_Ny, false);
    fits.set_ext_name("ORDER");
    fits.write_comment("Image order = max(0, floor(|phi_accumulated|/pi) - 1)");
    fits.write_comment("0 = direct image, 1 = first photon ring, -1 = captured by BH");
    fits.write_comment("Subtract 1 because backward-traced rays accumulate ~pi for direct image");
    write_axis_kw();

    // ESCAPED — binary escape mask.
    Array2D<double> escaped_out(img_Nx, img_Ny);
    for (int ix = 0; ix < img_Nx; ix++)
        for (int iy = 0; iy < img_Ny; iy++)
            escaped_out[ix][iy] = static_cast<double>(escaped_map[ix][iy]);
    fits.write_image(escaped_out, img_Nx, img_Ny, false);
    fits.set_ext_name("ESCAPED");
    fits.write_comment("1 = ray reached source sphere at r_lim; 0 = captured or lost");
    write_axis_kw();

    // THETA_S — source polar angle.
    fits.write_image(theta_s_map, img_Nx, img_Ny, false);
    fits.set_ext_name("THETA_S");
    fits.write_comment("Source polar angle at r_lim (radians); NaN if captured");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "RAD");
    write_axis_kw();

    // PHI_S — source azimuthal angle.
    fits.write_image(phi_s_map, img_Nx, img_Ny, false);
    fits.set_ext_name("PHI_S");
    fits.write_comment("Source azimuthal angle at r_lim (radians, [-pi,pi]); NaN if captured");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "RAD");
    write_axis_kw();

    // RDOT_FLIPS — radial turning-point count.
    Array2D<double> rdot_out(img_Nx, img_Ny);
    for (int ix = 0; ix < img_Nx; ix++)
        for (int iy = 0; iy < img_Ny; iy++)
            rdot_out[ix][iy] = static_cast<double>(rdot_map[ix][iy]);
    fits.write_image(rdot_out, img_Nx, img_Ny, false);
    fits.set_ext_name("RDOT_FLIPS");
    fits.write_comment("Radial turning-point count during propagation");
    fits.write_comment("Direct image: rdot_flips~1; n-th ring: rdot_flips~2n+1");
    write_axis_kw();

    // EQUAT_CROSS — equatorial crossing count.
    Array2D<double> equat_out(img_Nx, img_Ny);
    for (int ix = 0; ix < img_Nx; ix++)
        for (int iy = 0; iy < img_Ny; iy++)
            equat_out[ix][iy] = static_cast<double>(equat_map[ix][iy]);
    fits.write_image(equat_out, img_Nx, img_Ny, false);
    fits.set_ext_name("EQUAT_CROSS");
    fits.write_comment("Equatorial-plane (theta=pi/2) crossing count during propagation");
    write_axis_kw();

    fits.close();

    cout << "Done. Output: " << out_filename << endl;
    return 0;
}
