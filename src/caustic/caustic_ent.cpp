//
// caustic_ent.cpp
//
// Caustic / disc-plane intersection in the reverberation energy-time plane.
//
// Traces the curve(s) along which the caustic surface of the family of rays reaching a distant observer
// (see caustic_3d.cpp) meets the equatorial disc plane, and places every point of the curve in the
// energy/time plane of the reverberation response function computed by disc_ent_line_rbin ("ent plot"):
//
//   energy  E = E_line / g   with g = E_disc / E_obs the total Doppler + gravitational redshift of a photon
//                            emitted at that point of the disc (circular Keplerian orbit) and reaching the
//                            observer, from Raytracer::ray_redshift (reverse mode, V = -1) as in the imaging codes
//   time    t = (t_sd(r) + t_do - t_continuum) * rgs
//                            t_sd: coordinate time from the point source to the disc at radius r, binned in
//                                  annuli exactly as disc_ent_line_rbin does (or read from its DISC table)
//                            t_do: coordinate time from the disc to the observer's image plane (rays start at
//                                  t = 0 on the plane, so this is the centre ray's t at the crossing)
//                            t_continuum: direct source-to-observer time (continuum arrival), as in the ent code
//
// Method.  The 3D caustic condition J = det[dX/dx, dX/dy, dX/dtau] = 0 (caustic_bundle.h) reduces, on the
// plane theta = pi/2, to the vanishing of the 2D Jacobian of the disc landing map (x, y) -> (r, phi), so the
// caustic meets the disc exactly where J evaluated at the ray's equatorial crossing is zero.  For every
// image-plane pixel the bundle tracer lands the bundle exactly on theta = pi/2 at the ray's 1st .. max_eqcross-th
// equatorial crossing and records J there (DiscCrossing), giving one scalar image J_n(x, y) per crossing order
// n.  The caustic / disc curve of sheet n is the zero contour of J_n, extracted by marching squares
// (disc_caustic_contour.h) and linked into ordered polylines.  Each contour vertex, which lies on a grid edge
// between pixels of opposite sign(J_n), is then refined by bisection along the edge: new bundles are traced from
// the interpolated image-plane position up to their n-th equatorial crossing (refine_iter iterations); the
// converged bundle's centre ray gives r, phi, g and t_do at the caustic point directly.
//
// Adaptive refinement (refine_levels > 0): every cell of the plane with a sign change of J_n along one of its
// edges (for any sheet n) is split into four (adaptive_plane.h) and the new points traced, refine_levels
// times, so the contour is sampled at a spacing of dx / 2^refine_levels where it runs while the rest of the
// plane stays coarse (refine_status = 1 also splits cells whose corner rays differ in their number of
// equatorial crossings or fate, i.e. along the critical curve, which finds contour pieces narrower than a
// regular-grid pixel at a higher cost); the contours are then extracted on the leaf cells of the refined mesh.  Per-pixel images
// stay on the regular grid; the POINTS table lists every traced point.
//
// J_n is discontinuous (not zero) where the number of equatorial crossings of neighbouring rays changes; such
// sign changes are recognised after the refinement, when the crossing position still jumps by more than
// max_jump (rg) across the bracket of width dx / 2^refine_iter, and flagged ACCEPT = 0.
//
// FITS output:
//   primary header  -- run parameters (TSTART = t_continuum, LINEEN, RGS, ISCO, ENTFILE, REFLEV, ...)
//   CURVES          -- binary table, one row per contour vertex: SHEET (equatorial crossing order n), NCAUST
//                      (order N of the caustic meeting the disc here, as N in caustic_3d: 1 = primary caustic),
//                      CURVE, VERTEX, CLOSED,
//                      XIMG, YIMG (rg), R, PHI, X, Y (disc position; PHI in (-pi, pi]), G (E_disc/E_obs), ENERGY
//                      (E_line / G), TDO, TSD (rg/c), TIME (ent convention and units), INDISC (1 if rmin <= r <=
//                      rdisc), REFINED (1 if the bisection converged; 0: linear interpolation between the pixels),
//                      JSLOPE (+1 if J_n increases from pixel 0 to pixel 1 of the edge, -1 otherwise), ACCEPT (0 if the
//                      crossing position still jumps by JUMP > max_jump across the refined bracket: J_n is
//                      discontinuous there, not zero)
//   JDISC<n>, RDISC<n>, PHIDISC<n>, GDISC<n>, TDISC<n>, NCDISC<n>  -- per-pixel images for each sheet n: the
//                      unscaled Jacobian, landing r, phi, redshift, t_do and caustic count at the n-th equatorial
//                      crossing (NaN where the ray has no n-th crossing)
//   NEQCROSS, STATUS -- per-pixel images: total equatorial crossings; termination status (as caustic_3d, plus
//                      6 = stopped after max_eqcross crossings)
//   POINTS          -- table (only with refine_levels > 0): XIMG, YIMG, LEVEL, NEQCROSS, STATUS of every traced point
//   DISC            -- table: r, time used for t_sd(r) (copied from the ent file or computed here)
//
// Array dimensions: the user specifies Nx, Ny grid steps; the image plane has (Nx+1)*(Ny+1) rays (fencepost
// convention) and all image extensions have dimensions Nx+1, Ny+1.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdio>

using namespace std;

#include "../raytracer/imageplane.h"
#include "../raytracer/pointsource.h"
#include "caustic_bundle.h"
#include "disc_caustic_contour.h"
#include "single_ray_bundle.h"
#include "adaptive_plane.h"

#include "../include/fits_output.h"
#include "../include/array.h"
#include "../include/par_file.h"
#include "../include/par_args.h"
#include "../include/kerr.h"
#include "../include/progress_bar.h"

#include "fitsio.h"

static const double NaN = numeric_limits<double>::quiet_NaN();

// ------------------------------------------------------------------------------------------------------------
// source-to-disc light travel time per annulus (the DISC table of disc_ent_line_rbin)
// ------------------------------------------------------------------------------------------------------------
struct SourceTimeTable
{
    vector<double> r, t;       // annulus start radius and mean arrival time (NaN if no rays)
    vector<int> count;
    double r_min, r_disc, dr;
    bool logbin;
    int Nr;

    double r_mid(int i) const { return logbin ? r[i] * sqrt(dr) : r[i] + 0.5 * dr; }
    double r_end(int i) const { return logbin ? r[i] * dr : r[i] + dr; }

    // t_sd(r): linear interpolation between annulus mid-points (interp = true) or the ent code's per-annulus
    // step function (interp = false); NaN outside [r_min, r_disc]
    double time(double rr, bool interp) const
    {
        if (!(rr >= r_min && rr <= r_disc) || Nr == 0) return NaN;
        int ir = logbin ? static_cast<int>(log(rr / r_min) / log(dr)) : static_cast<int>((rr - r_min) / dr);
        ir = max(0, min(Nr - 1, ir));
        if (!interp) return t[ir];
        // nearest defined bins on either side of rr
        int lo = -1, hi = -1;
        for (int i = ir; i >= 0; i--) if (isfinite(t[i]) && r_mid(i) <= rr) { lo = i; break; }
        for (int i = ir; i < Nr; i++)  if (isfinite(t[i]) && r_mid(i) >= rr) { hi = i; break; }
        if (lo < 0 && hi < 0) return NaN;
        if (lo < 0) return t[hi];
        if (hi < 0) return t[lo];
        if (lo == hi) return t[lo];
        const double f = (rr - r_mid(lo)) / (r_mid(hi) - r_mid(lo));
        return t[lo] + f * (t[hi] - t[lo]);
    }
};

static void fits_check(int status, const string& what)
{
    if (status)
    {
        char msg[81];
        fits_get_errstatus(status, msg);
        cerr << "cfitsio error (" << what << "): " << msg << endl;
        exit(1);
    }
}

// read the DISC table (r, time, count) and the relevant header keywords from a disc_ent_line_rbin file
static SourceTimeTable read_ent_disc_table(const string& filename, double& spin, double& line_en,
                                           double& t_continuum, double& rgs, double& r_isco, bool& have_rgs)
{
    SourceTimeTable T;
    fitsfile* f; int status = 0;
    fits_open_file(&f, filename.c_str(), READONLY, &status);
    fits_check(status, "open " + filename);

    auto read_key = [&](const char* key, double& val, bool required) -> bool
    {
        int st = 0; char comment[81];
        fits_read_key(f, TDOUBLE, key, &val, comment, &st);
        if (st && required) fits_check(st, string("read keyword ") + key);
        return st == 0;
    };
    read_key("SPIN", spin, true);
    read_key("LINEEN", line_en, true);
    read_key("ISCO", r_isco, true);
    have_rgs = read_key("RGS", rgs, false);
    if (!have_rgs)
    {
        double mass = -1; read_key("MASS", mass, false);
        rgs = (mass > 0) ? 6.67E-11 * mass * (2E30) / pow(3E8, 3) : 1.0;
        have_rgs = true;
    }
    // TSTART lives in the RESPONSE image header
    fits_movnam_hdu(f, ANY_HDU, (char*) "RESPONSE", 0, &status);
    fits_check(status, "move to RESPONSE");
    read_key("TSTART", t_continuum, true);

    fits_movnam_hdu(f, BINARY_TBL, (char*) "DISC", 0, &status);
    fits_check(status, "move to DISC");
    long rows; fits_get_num_rows(f, &rows, &status); fits_check(status, "DISC rows");
    T.Nr = static_cast<int>(rows);
    T.r.resize(rows); T.t.resize(rows); T.count.resize(rows);
    int col; int anynul = 0;
    fits_get_colnum(f, CASEINSEN, (char*) "r", &col, &status);
    fits_read_col(f, TDOUBLE, col, 1, 1, rows, nullptr, T.r.data(), &anynul, &status);
    fits_get_colnum(f, CASEINSEN, (char*) "time", &col, &status);
    fits_read_col(f, TDOUBLE, col, 1, 1, rows, nullptr, T.t.data(), &anynul, &status);
    fits_get_colnum(f, CASEINSEN, (char*) "count", &col, &status);
    fits_read_col(f, TINT, col, 1, 1, rows, nullptr, T.count.data(), &anynul, &status);
    fits_check(status, "read DISC columns");
    fits_close_file(f, &status);

    for (int i = 0; i < T.Nr; i++) if (T.count[i] == 0 || !isfinite(T.t[i])) T.t[i] = NaN;
    T.r_min = T.r[0];
    T.logbin = (T.Nr > 2) ? (fabs((T.r[2] / T.r[1]) - (T.r[1] / T.r[0])) < fabs((T.r[2] - T.r[1]) - (T.r[1] - T.r[0]))) : false;
    T.dr = T.logbin ? T.r[1] / T.r[0] : T.r[1] - T.r[0];
    T.r_disc = T.r_end(T.Nr - 1);
    return T;
}

// trace the point source to the disc and bin the arrival times in annuli, as disc_ent_line_rbin does
static SourceTimeTable compute_source_times(double source[4], double V, double spin, double cosalpha0, double cosalphamax,
                                            double dcosalpha, double beta0, double betamax, double dbeta, double r_max,
                                            double r_min, double r_disc, int Nr, bool logbin_r, Integrator integrator,
                                            double rk45_tol, double symp_step, int symp_order, double max_tstep,
                                            int show_progress)
{
    SourceTimeTable T;
    T.Nr = Nr; T.logbin = logbin_r; T.r_min = r_min; T.r_disc = r_disc;
    T.dr = logbin_r ? exp(log(r_disc / r_min) / Nr) : (r_disc - r_min) / Nr;
    T.r.resize(Nr); T.t.assign(Nr, 0); T.count.assign(Nr, 0);
    for (int ir = 0; ir < Nr; ir++) T.r[ir] = logbin_r ? r_min * pow(T.dr, ir) : r_min + ir * T.dr;

    cout << "Tracing the point source to the disc for the source-to-disc times..." << endl;
    PointSource<double>* src = new PointSource<double>(source, V, spin, TOL, dcosalpha, dbeta, cosalpha0, cosalphamax, beta0, betamax);
    src->set_rk45_tol(rk45_tol);
    if (symp_step > 0) src->set_symplectic_step(symp_step);
    src->set_symplectic_order(symp_order);
    src->set_max_tstep(max_tstep);
    src->redshift_start();
    src->run_raytrace(integrator, M_PI_2, r_max, show_progress);
    src->range_phi();

    long disc_count = 0;
    for (int ray = 0; ray < src->get_count(); ray++)
    {
        if (src->rays[ray].status <= 0) continue;
        double cx, cy, cz;
        cartesian<double>(cx, cy, cz, src->rays[ray].r, src->rays[ray].theta, src->rays[ray].phi, spin);
        if (cz < 1E-2 && src->rays[ray].r <= r_disc && src->rays[ray].r >= r_min)
        {
            const int ir = logbin_r ? static_cast<int>(log(src->rays[ray].r / r_min) / log(T.dr))
                                    : static_cast<int>((src->rays[ray].r - r_min) / T.dr);
            if (ir >= 0 && ir < Nr)
            {
                ++T.count[ir];
                T.t[ir] += src->rays[ray].t;
            }
            ++disc_count;
        }
    }
    delete src;
    cout << disc_count << " photons hit the disc" << endl;
    for (int ir = 0; ir < Nr; ir++) T.t[ir] = (T.count[ir] > 0) ? T.t[ir] / T.count[ir] : NaN;
    return T;
}

struct SheetPixel
{
    bool valid = false;
    DiscCrossing dc;
    double J = NaN, g = NaN, X = NaN, Y = NaN;
};

struct CurveRow
{
    int sheet, ncaust, curve, vertex, closed;
    double ximg, yimg, r, phi, X, Y, g, energy, tdo, tsd, time;
    int indisc, refined, jslope, accept;
    double jump;
};

int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile")
                          ? par_args.get_parameter<string>("--parfile") : string("../par/caustic_ent.par");
    ParameterFile par_file(par_filename);

    string out_filename = par_args.key_exists("--outfile")
                          ? par_args.get_parameter<string>("--outfile")
                          : par_file.get_parameter<string>("outfile");
    string ent_filename = par_args.key_exists("--entfile")
                          ? par_args.get_parameter<string>("--entfile")
                          : par_file.get_parameter<string>("entfile", "");

    // --- observer / image plane / bundle (as caustic_3d) ---
    double dist  = par_file.get_parameter<double>("dist");
    double incl  = par_args.key_exists("--incl") ? par_args.get_parameter<double>("--incl") : par_file.get_parameter<double>("incl");
    double plane_phi0 = par_file.get_parameter<double>("plane_phi0", 0);
    double spin  = par_args.key_exists("--spin") ? par_args.get_parameter<double>("--spin") : par_file.get_parameter<double>("spin");
    double x0   = par_file.get_parameter<double>("x0");
    double xmax = par_file.get_parameter<double>("xmax");
    int    Nx   = par_args.key_exists("--Nx") ? par_args.get_parameter<int>("--Nx") : par_file.get_parameter<int>("Nx");
    double y0   = par_file.get_parameter<double>("y0",   x0);
    double ymax = par_file.get_parameter<double>("ymax", xmax);
    int    Ny   = par_args.key_exists("--Ny") ? par_args.get_parameter<int>("--Ny") : par_file.get_parameter<int>("Ny", Nx);
    double delta = par_args.key_exists("--delta") ? par_args.get_parameter<double>("--delta") : par_file.get_parameter<double>("delta", 1e-4);
    double delta_max = par_file.get_parameter<double>("delta_max", 100*delta);
    double r_max     = par_file.get_parameter<double>("r_max", 1.1 * dist);
    double symp_step   = par_file.get_parameter<double>("symp_step", -1);
    int    symp_order  = par_file.get_parameter<int>("symp_order", 6);
    double max_tstep   = par_file.get_parameter<double>("max_tstep", MAXDT);
    double maxtstep_rlim = par_file.get_parameter<double>("maxtstep_rlim", MAXDT_RLIM);
    double max_phistep = par_file.get_parameter<double>("max_phistep", MAXDPHI);
    double precision   = par_file.get_parameter<double>("precision", PRECISION);
    int    steplim     = par_file.get_parameter<int>("steplim", SYMP_STEPLIM);
    int    max_eqcross = par_file.get_parameter<int>("max_eqcross", 3);
    int    refine_iter = par_file.get_parameter<int>("refine_iter", 10);
    double max_jump    = par_file.get_parameter<double>("max_jump", 2.0);
    int    refine_levels = par_args.key_exists("--refine_levels") ? par_args.get_parameter<int>("--refine_levels")
                                                                  : par_file.get_parameter<int>("refine_levels", 0);
    bool   refine_status = par_file.get_parameter<bool>("refine_status", false);   // also refine where the rays' fate / crossing count changes
    int show_progress  = par_args.key_exists("--show_progress") ? par_args.get_parameter<int>("--show_progress")
                                                                : par_file.get_parameter<int>("show_progress", 1);

    // --- source-to-disc times (as disc_ent_line_rbin) ---
    double source[4] = { 0, 5, 1e-3, 0 };
    double V = par_file.get_parameter<double>("V", 0);
    double cosalpha0 = par_file.get_parameter<double>("cosalpha0", -0.9999);
    double cosalphamax = par_file.get_parameter<double>("cosalphamax", 0.9999);
    double dcosalpha = par_file.get_parameter<double>("dcosalpha", 1e-3);
    double beta0 = par_file.get_parameter<double>("beta0", -1 * M_PI);
    double betamax = par_file.get_parameter<double>("betamax", M_PI);
    double dbeta = par_file.get_parameter<double>("dbeta", 1e-2);
    double src_rmax = par_file.get_parameter<double>("rmax", 1000);
    double r_disc = par_file.get_parameter<double>("rdisc", 500);
    double r_min = par_file.get_parameter<double>("rmin", -1);
    int Nr = par_file.get_parameter<int>("Nr", 100);
    bool logbin_r = par_file.get_parameter<bool>("logbin_r", true);
    double line_en = par_file.get_parameter<double>("line_en", 6.4);
    double t_continuum = par_args.key_exists("--t_continuum") ? par_args.get_parameter<double>("--t_continuum")
                                                              : par_file.get_parameter<double>("t_continuum", 0);
    double mass = par_file.get_parameter<double>("mass", -1);
    bool time_interp = par_file.get_parameter<bool>("time_interp", true);
    string integrator_str = par_file.get_parameter<string>("integrator", "symplectic");
    double rk45_tol = par_file.get_parameter<double>("rk45_tol", 1e-8);

    Integrator integrator = Integrator::RK45;
    if (integrator_str == "euler") integrator = Integrator::Euler;
    else if (integrator_str == "rk4") integrator = Integrator::RK4;
    else if (integrator_str == "symplectic") integrator = Integrator::Symplectic;

    double rgs = (mass > 0) ? 6.67E-11 * mass * (2E30) / pow(3E8, 3) : 1.0;
    double r_isco = kerr_isco<double>(spin, +1);
    const double r_horizon = kerr_horizon<double>(spin);

    // --- source-to-disc time table ---
    SourceTimeTable stt;
    if (!ent_filename.empty())
    {
        double ent_spin, ent_line_en, ent_tcont, ent_rgs, ent_isco; bool have_rgs;
        stt = read_ent_disc_table(ent_filename, ent_spin, ent_line_en, ent_tcont, ent_rgs, ent_isco, have_rgs);
        cout << "Source-to-disc times from " << ent_filename << ": " << stt.Nr << " annuli, r = " << stt.r_min
             << " to " << stt.r_disc << (stt.logbin ? " (log)" : " (linear)") << endl;
        cout << "  header: spin = " << ent_spin << ", line_en = " << ent_line_en << ", t_continuum = " << ent_tcont
             << ", rgs = " << ent_rgs << endl;
        if (fabs(ent_spin - spin) > 1e-6)
            cerr << "WARNING: spin in " << ent_filename << " (" << ent_spin << ") differs from the par file (" << spin << ")" << endl;
        line_en = ent_line_en; t_continuum = ent_tcont; rgs = ent_rgs;
        r_min = stt.r_min; r_disc = stt.r_disc;
    }
    else
    {
        par_file.get_parameter_array("source", source, 4);
        if (par_args.key_exists("--source_h")) source[1] = par_args.get_parameter<double>("--source_h");
        if (r_min < 0) r_min = r_isco;
        stt = compute_source_times(source, V, spin, cosalpha0, cosalphamax, dcosalpha, beta0, betamax, dbeta, src_rmax,
                                   r_min, r_disc, Nr, logbin_r, integrator, rk45_tol, symp_step, symp_order, max_tstep,
                                   show_progress);
    }

    // --- image plane ---
    const double dx = (xmax - x0) / Nx, dy = (ymax - y0) / Ny;
    const double dx_ip = dx * (1 - 1e-9), dy_ip = dy * (1 - 1e-9);
    const int img_Nx = Nx + 1, img_Ny = Ny + 1, nPix = img_Nx * img_Ny;

    cout << "*****" << endl;
    cout << "Observer at r = " << dist << ", incl = " << incl << " deg, phi0 = " << plane_phi0 << endl;
    cout << "Spin a = " << spin << "  (horizon " << r_horizon << ", ISCO " << r_isco << ")" << endl;
    cout << "Disc r = " << r_min << " to " << r_disc << "; line energy " << line_en << "; t_continuum = " << t_continuum << endl;
    cout << "Image plane: " << img_Nx << " x " << img_Ny << " = " << nPix << " bundles of 5 rays; "
         << max_eqcross << " equatorial crossings per ray" << endl;

    const double offx[5] = { 0, delta, -delta, 0, 0 }, offy[5] = { 0, 0, 0, delta, -delta };
    ImagePlane<double>* planes[5];
    for (int i = 0; i < 5; i++)
    {
        planes[i] = new ImagePlane<double>(dist, incl, x0 + offx[i], xmax + offx[i], dx_ip,
                                           y0 + offy[i], ymax + offy[i], dy_ip, spin, plane_phi0, precision);
        if (planes[i]->get_count() != nPix)
        {
            cerr << "Error: image plane " << i << " has " << planes[i]->get_count() << " rays, expected " << nPix << endl;
            return 1;
        }
    }
    planes[0]->redshift_start();

    BundleParams P;
    P.a = -spin;
    P.horizon = r_horizon;
    P.h0 = (symp_step > 0) ? symp_step : 1.0 / precision;
    P.r_cap = 2 * r_horizon;
    P.max_tstep = max_tstep; P.maxtstep_rlim = maxtstep_rlim; P.max_phistep = max_phistep;
    P.r_max = r_max;
    P.delta = delta; P.delta_max = delta_max;
    P.order = symp_order;
    P.steplim = steplim;
    P.max_caustics = 0;
    P.max_eqcross = max_eqcross;
    P.stop_after_eqcross = max_eqcross;

    const Ray<double>* ray_arrays[5];
    for (int i = 0; i < 5; i++) ray_arrays[i] = planes[i]->rays;

    // --- per-pixel bundles ---
    vector<PixelResult> results(nPix);
    vector<vector<SheetPixel>> sheets(max_eqcross, vector<SheetPixel>(nPix));

    // evaluate the redshift and Cartesian disc position of a crossing
    auto fill_sheet_pixel = [&](SheetPixel& sp, const DiscCrossing& dc, double emit)
    {
        sp.valid = true;
        sp.dc = dc;
        sp.J = dc.J * pow(10.0, dc.logscale);
        sp.g = (dc.r >= r_isco) ? disc_crossing_redshift(*planes[0], dc, P.a, emit) : NaN;
        double Z;
        cartesian<double>(sp.X, sp.Y, Z, dc.r, M_PI_2, dc.phi, spin);
    };

    cout << "Tracing image-plane bundles to their equatorial crossings..." << endl;
    ProgressBar prog(nPix, "Bundle", 0, (show_progress > 0));
    int pix_done = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int pix = 0; pix < nPix; pix++)
    {
        if (show_progress != 0)
        {
            int done;
            #pragma omp atomic capture
            done = ++pix_done;
            if (done % show_progress == 0)
            {
                #pragma omp critical
                prog.show(done);
            }
        }
        BundleRay b[5];
        PixelResult& res = results[pix];
        if (!bundle_from_rays(ray_arrays, pix, P.a, b))
        {
            res.status = STATUS_SKIPPED; res.ncaust = 0; res.eqcross = 0; res.tau_end = 0;
            continue;
        }
        vector<CausticPoint> dummy; vector<DiscCrossing> dc;
        trace_bundle(b, P, planes[0]->get_x_index(pix), planes[0]->get_y_index(pix),
                     planes[0]->rays[pix].alpha, planes[0]->rays[pix].beta, dummy, res, nullptr, &dc);
        for (auto& d : dc)
            if (d.n >= 1 && d.n <= max_eqcross) fill_sheet_pixel(sheets[d.n - 1][pix], d, planes[0]->rays[pix].emit);
    }
    prog.done();

    // observer-frame energy of the centre ray of every point (regular grid: from redshift_start())
    vector<double> emits(nPix);
    for (int pix = 0; pix < nPix; pix++) emits[pix] = planes[0]->rays[pix].emit;

    // --- adaptive refinement of the plane around the sign changes of J_n ---
    AdaptivePlane aplane(x0, y0, dx, dy, Nx, Ny, refine_levels);
    if (refine_levels > 0)
    {
        cout << "Adaptive refinement: " << refine_levels << " level(s), finest spacing "
             << aplane.fine_dx() << " x " << aplane.fine_dy() << " rg" << endl;
        for (int level = 0; level < refine_levels; level++)
        {
            vector<int> created = aplane.refine(level, [&](const PlaneCell& c)
            {
                if (refine_status)
                    for (int e = 1; e < 4; e++)
                        if (results[c.corner[e]].eqcross != results[c.corner[0]].eqcross
                            || results[c.corner[e]].status != results[c.corner[0]].status) return true;
                for (int n = 0; n < max_eqcross; n++)
                    for (int e = 0; e < 4; e++)
                    {
                        const SheetPixel& p = sheets[n][c.corner[e]];
                        const SheetPixel& q = sheets[n][c.corner[(e + 1) % 4]];
                        if (p.valid && q.valid && isfinite(p.J) && isfinite(q.J) && ((p.J >= 0) != (q.J >= 0))) return true;
                    }
                return false;
            });
            results.resize(aplane.npoints());
            emits.resize(aplane.npoints());
            for (int n = 0; n < max_eqcross; n++) sheets[n].resize(aplane.npoints());
            cout << "  level " << level + 1 << ": " << created.size() << " new points" << endl;
            if (created.empty()) break;

            ProgressBar rprog(static_cast<long>(created.size()), "Point", 0, (show_progress > 0));
            int done_pts = 0;
            #pragma omp parallel
            {
                SingleRayBundle* srb;
                #pragma omp critical
                srb = new SingleRayBundle(dist, incl, plane_phi0, spin, delta, precision);

                #pragma omp for schedule(dynamic)
                for (size_t m = 0; m < created.size(); m++)
                {
                    if (show_progress != 0)
                    {
                        int done;
                        #pragma omp atomic capture
                        done = ++done_pts;
                        if (done % show_progress == 0)
                        {
                            #pragma omp critical
                            rprog.show(done);
                        }
                    }
                    const int k = created[m];
                    const PlanePoint& pt = aplane.point(k);
                    PixelResult& res = results[k];
                    BundleRay b[5];
                    if (!srb->init(pt.x, pt.y, b, &emits[k]))
                    {
                        res.status = STATUS_SKIPPED; res.ncaust = 0; res.eqcross = 0; res.tau_end = 0;
                        continue;
                    }
                    vector<CausticPoint> dummy; vector<DiscCrossing> dc;
                    trace_bundle(b, P, static_cast<int>(pt.i), static_cast<int>(pt.j), pt.x, pt.y, dummy, res, nullptr, &dc);
                    for (auto& d : dc)
                        if (d.n >= 1 && d.n <= max_eqcross) fill_sheet_pixel(sheets[d.n - 1][k], d, emits[k]);
                }
                #pragma omp critical
                delete srb;
            }
            rprog.done();
        }
        cout << "  " << aplane.npoints() << " points traced in total (" << nPix << " on the regular grid)" << endl;
    }
    const vector<PlaneCell> leaf_cells = aplane.leaf_cells();
    const vector<PlanePoint>& pts = aplane.points();

    // --- contours per sheet on the leaf cells, refined along the cell edges ---

    vector<CurveRow> rows;
    int total_rejected = 0, total_unrefined = 0;

    for (int n = 1; n <= max_eqcross; n++)
    {
        vector<SheetPixel>& S = sheets[n - 1];
        vector<double> field(aplane.npoints());
        for (int k = 0; k < aplane.npoints(); k++) field[k] = S[k].valid ? S[k].J : NaN;

        vector<ContourCurve> curves = zero_contours_cells(leaf_cells, field, pts);
        size_t nverts = 0; for (auto& c : curves) nverts += c.vertices.size();
        cout << "Sheet " << n << ": " << curves.size() << " curve(s), " << nverts << " vertices" << endl;

        // flatten the vertices for parallel refinement
        struct VRef { int curve, vertex; };
        vector<VRef> vrefs;
        for (size_t c = 0; c < curves.size(); c++)
            for (size_t v = 0; v < curves[c].vertices.size(); v++) vrefs.push_back({static_cast<int>(c), static_cast<int>(v)});
        vector<CurveRow> sheet_rows(vrefs.size());

        BundleParams Pn = P;
        Pn.max_eqcross = n; Pn.stop_after_eqcross = n;

        ProgressBar rprog(static_cast<long>(vrefs.size()), "Vertex", 0, (show_progress > 0 && refine_iter > 0));
        int v_done = 0, unrefined = 0, rejected = 0;
        #pragma omp parallel
        {
            SingleRayBundle* srb = nullptr;
            if (refine_iter > 0)
            {
                #pragma omp critical
                srb = new SingleRayBundle(dist, incl, plane_phi0, spin, delta, precision);
            }

            // trace a bundle from (x, y) to its n-th equatorial crossing; returns false if it has none
            auto eval = [&](double x, double y, DiscCrossing& out) -> bool
            {
                BundleRay b[5];
                if (!srb->init(x, y, b)) return false;
                vector<CausticPoint> dummy; vector<DiscCrossing> dc; PixelResult r;
                trace_bundle(b, Pn, 0, 0, x, y, dummy, r, nullptr, &dc);
                for (auto& d : dc) if (d.n == n && isfinite(d.J)) { out = d; return true; }
                return false;
            };

            #pragma omp for schedule(dynamic)
            for (size_t k = 0; k < vrefs.size(); k++)
            {
                if (show_progress != 0 && refine_iter > 0)
                {
                    int done;
                    #pragma omp atomic capture
                    done = ++v_done;
                    if (done % show_progress == 0)
                    {
                        #pragma omp critical
                        rprog.show(done);
                    }
                }
                const ContourCurve& c = curves[vrefs[k].curve];
                const ContourVertex& v = c.vertices[vrefs[k].vertex];
                const SheetPixel& p0 = S[v.p0];
                const SheetPixel& p1 = S[v.p1];
                const double xa = pts[v.p0].x, ya = pts[v.p0].y;
                const double xb = pts[v.p1].x, yb = pts[v.p1].y;

                CurveRow row;
                row.sheet = n; row.curve = vrefs[k].curve; row.vertex = vrefs[k].vertex; row.closed = c.closed ? 1 : 0;
                row.jslope = (p1.J > p0.J) ? 1 : -1;

                // bisection along the edge on the sign of J_n, keeping the crossings at both ends of the bracket
                double lo = 0, hi = 1;
                DiscCrossing d_lo = p0.dc, d_hi = p1.dc, best;
                double Jlo = p0.J;
                bool have_best = false; bool converged = (refine_iter > 0);
                for (int it = 0; it < refine_iter; it++)
                {
                    const double s = 0.5 * (lo + hi);
                    DiscCrossing d;
                    if (!eval(xa + s * (xb - xa), ya + s * (yb - ya), d)) { converged = false; break; }
                    const double J = d.J * pow(10.0, d.logscale);
                    if ((J >= 0) == (Jlo >= 0)) { lo = s; Jlo = J; d_lo = d; } else { hi = s; d_hi = d; }
                    best = d; have_best = true;
                }
                double s_final = 0.5 * (lo + hi);
                if (have_best && converged)
                {
                    // final evaluation at the bracket mid-point
                    DiscCrossing d;
                    if (eval(xa + s_final * (xb - xa), ya + s_final * (yb - ya), d)) best = d; else converged = false;
                }
                row.refined = (have_best && converged) ? 1 : 0;
                // residual jump of the crossing position across the bracket: small for a genuine zero of a
                // continuous J_n (even at a fold, where the position varies as sqrt of the distance to the
                // caustic), finite where J_n is discontinuous (the n-th crossing of neighbouring rays jumps)
                {
                    double Xl, Yl, Xh, Yh, Z;
                    cartesian<double>(Xl, Yl, Z, d_lo.r, M_PI_2, d_lo.phi, spin);
                    cartesian<double>(Xh, Yh, Z, d_hi.r, M_PI_2, d_hi.phi, spin);
                    row.jump = hypot(Xl - Xh, Yl - Yh);
                    row.accept = (row.jump <= max_jump) ? 1 : 0;
                    if (!row.accept)
                    {
                        #pragma omp atomic
                        ++rejected;
                    }
                }

                if (row.refined)
                {
                    const double emit = emits[v.p0] + s_final * (emits[v.p1] - emits[v.p0]);
                    row.ximg = xa + s_final * (xb - xa); row.yimg = ya + s_final * (yb - ya);
                    row.r = best.r; row.phi = best.phi; row.tdo = best.t;
                    // order of the caustic meeting the disc here: the rays on one side of the bracket cross it
                    // before reaching the disc, those on the other side after
                    row.ncaust = min(d_lo.ncaust, d_hi.ncaust) + 1;
                    double Z; cartesian<double>(row.X, row.Y, Z, best.r, M_PI_2, best.phi, spin);
                    row.g = (best.r >= r_isco) ? disc_crossing_redshift(*planes[0], best, P.a, emit) : NaN;
                }
                else
                {
                    // linear interpolation between the two pixels (position in Cartesian disc coordinates)
                    const double f = v.f;
                    row.ximg = v.x; row.yimg = v.y;
                    row.X = p0.X + f * (p1.X - p0.X); row.Y = p0.Y + f * (p1.Y - p0.Y);
                    row.r = hypot(row.X, row.Y); row.phi = atan2(row.Y, row.X);
                    row.tdo = p0.dc.t + f * (p1.dc.t - p0.dc.t);
                    row.g = p0.g + f * (p1.g - p0.g);
                    row.ncaust = min(p0.dc.ncaust, p1.dc.ncaust) + 1;
                    #pragma omp atomic
                    ++unrefined;
                }
                row.indisc = (row.r >= r_min && row.r <= r_disc) ? 1 : 0;
                row.tsd = stt.time(row.r, time_interp);
                row.energy = line_en / row.g;
                row.time = (row.tsd + row.tdo - t_continuum) * rgs;
                sheet_rows[k] = row;
            }
            #pragma omp critical
            delete srb;     // (the destructor also redirects cout, so one thread at a time)
        }
        rprog.done();
        total_unrefined += unrefined;
        total_rejected += rejected;
        if (refine_iter > 0 && unrefined > 0)
            cout << "  " << unrefined << " vertex/vertices could not be refined (linear interpolation used)" << endl;
        cout << "  " << rejected << " vertex/vertices rejected by the jump test (crossing position discontinuous)" << endl;
        rows.insert(rows.end(), sheet_rows.begin(), sheet_rows.end());
    }

    cout << rows.size() << " caustic / disc-plane points in total" << endl;

    // ------------------------------------------------------------------------------------------------
    // FITS output
    // ------------------------------------------------------------------------------------------------
    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Caustic / disc-plane intersection in the reverberation energy-time plane");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "caustic_ent");
    fits.write_keyword("DIST",    "Observer distance / image plane distance (rg)", dist);
    fits.write_keyword("INCL",    "Observer inclination (degrees from spin axis)", incl);
    fits.write_keyword("PHI0",    "Azimuth of the image plane centre (radians)", plane_phi0);
    fits.write_keyword("SPIN",    "Black hole spin parameter a/M", spin);
    fits.write_keyword("HORIZON", "Event horizon radius (rg)", r_horizon);
    fits.write_keyword("ISCO",    "Innermost stable circular orbit (rg)", r_isco);
    fits.write_keyword("RMIN",    "Inner disc radius (rg)", r_min);
    fits.write_keyword("RDISC",   "Outer disc radius (rg)", r_disc);
    fits.write_keyword("LINEEN",  "Rest frame energy of emission line", line_en);
    fits.write_keyword("TSTART",  "Continuum arrival time subtracted from all times (t_continuum)", t_continuum);
    fits.write_keyword("RGS",     "Gravitational units to seconds conversion applied to TIME", rgs);
    fits.write_keyword("MASS",    "Black hole mass (M_sun; <0: natural units)", mass);
    fits.write_keyword("TUNIT",   "Units of TIME column", (rgs != 1.0) ? "SECOND" : "GM/c^3");
    fits.write_keyword("ENTFILE", "ent file the source-to-disc times were read from ('' = computed)", ent_filename);
    if (ent_filename.empty())
    {
        fits.write_keyword("SRC_T", "Source t co-ordinate", source[0]);
        fits.write_keyword("SRC_R", "Source r co-ordinate", source[1]);
        fits.write_keyword("SRC_TH", "Source theta co-ordinate", source[2]);
        fits.write_keyword("SRC_PHI", "Source phi co-ordinate", source[3]);
        fits.write_keyword("V", "Source angular velocity d(phi)/dt", V);
        fits.write_keyword("DCOSALPH", "Step in cos source polar angle", dcosalpha);
        fits.write_keyword("DBETA", "Step in source azimuthal angle", dbeta);
    }
    fits.write_keyword("TINTERP", "Source-to-disc time interpolated between annuli", time_interp);
    fits.write_keyword("DELTA",   "Bundle offset in the image plane (rg)", delta);
    fits.write_keyword("DELTAMAX", "Bundle rescaling threshold (rg)", delta_max);
    fits.write_keyword("RMAX",    "Escape radius (rg)", r_max);
    fits.write_keyword("SYMPSTEP", "Mino-time step in the strong field", P.h0);
    fits.write_keyword("SYMPORD", "Symplectic composition order", symp_order);
    fits.write_keyword("MAXTSTEP", "Far-field cap on the coordinate-time step", max_tstep);
    fits.write_keyword("MAXEQCR", "Equatorial crossings recorded per ray (sheets)", max_eqcross);
    fits.write_keyword("REFITER", "Bisection iterations per contour vertex", refine_iter);
    fits.write_keyword("MAXJUMP", "Max crossing-position jump across a refined bracket for ACCEPT (rg)", max_jump);
    fits.write_keyword("NREJECT", "Vertices with ACCEPT = 0", total_rejected);
    fits.write_keyword("REFLEV",  "Adaptive refinement levels of the image plane", refine_levels);
    fits.write_keyword("REFSTAT", "Refinement also on crossing-count / fate changes", refine_status);
    fits.write_keyword("NPTS",    "Image-plane points traced (regular grid + refined)", static_cast<long>(results.size()));
    fits.write_keyword("NUNREF",  "Vertices not refined (interpolated)", total_unrefined);
    fits.write_keyword("NX",   "Number of pixels in X", img_Nx);
    fits.write_keyword("NY",   "Number of pixels in Y", img_Ny);
    fits.write_keyword("X0",   "Start of X axis (rg)", x0);
    fits.write_keyword("XMAX", "End of X axis (rg)", xmax);
    fits.write_keyword("Y0",   "Start of Y axis (rg)", y0);
    fits.write_keyword("YMAX", "End of Y axis (rg)", ymax);
    fits.write_keyword("NPOINTS", "Number of caustic / disc-plane points stored", static_cast<long>(rows.size()));

    // CURVES table
    {
        const long nrows = static_cast<long>(rows.size());
        const int ncols = 21;
        const char* names[ncols]   = { "SHEET", "NCAUST", "CURVE", "VERTEX", "CLOSED", "XIMG", "YIMG", "R", "PHI", "X", "Y",
                                       "G", "ENERGY", "TDO", "TSD", "TIME", "INDISC", "REFINED", "JSLOPE", "ACCEPT", "JUMP" };
        const char* formats[ncols] = { "1J", "1J", "1J", "1J", "1J", "1D", "1D", "1D", "1D", "1D", "1D",
                                       "1D", "1D", "1D", "1D", "1D", "1J", "1J", "1J", "1J", "1D" };
        const char* units[ncols]   = { "", "", "", "", "", "rg", "rg", "rg", "rad", "rg", "rg",
                                       "", "", "rg/c", "rg/c", (rgs != 1.0) ? "s" : "rg/c", "", "", "", "", "rg" };
        char* c_names[ncols]; char* c_formats[ncols]; char* c_units[ncols];
        for (int c = 0; c < ncols; c++)
        {
            c_names[c] = strdup(names[c]); c_formats[c] = strdup(formats[c]); c_units[c] = strdup(units[c]);
        }
        char extname[] = "CURVES";
        fits.create_table(extname, ncols, nrows, c_names, c_formats, c_units);
        fits.write_comment("One row per vertex of the caustic / disc-plane curves, ordered along each curve");
        fits.write_comment("SHEET: equatorial crossing order n of the ray; NCAUST: order N of the caustic meeting the disc (1 = primary)");
        fits.write_comment("G = E_disc/E_obs; ENERGY = LINEEN/G; TIME = (TSD + TDO - TSTART) * RGS");
        fits.write_comment("ACCEPT = 0: the crossing position jumps by JUMP > MAXJUMP across the refined bracket (J discontinuous)");
        fits.write_keyword("TSTART", "Continuum arrival time subtracted", t_continuum);
        fits.write_keyword("LINEEN", "Rest frame line energy", line_en);
        fits.write_keyword("RGS", "Time conversion applied to TIME", rgs);

        vector<int> icol(nrows); vector<double> dcol(nrows);
        auto write_int = [&](int (*get)(const CurveRow&)) { for (long i = 0; i < nrows; i++) icol[i] = get(rows[i]); fits.write_table_column(icol.data(), nrows); };
        auto write_dbl = [&](double (*get)(const CurveRow&)) { for (long i = 0; i < nrows; i++) dcol[i] = get(rows[i]); fits.write_table_column(dcol.data(), nrows); };
        write_int([](const CurveRow& r) { return r.sheet; });
        write_int([](const CurveRow& r) { return r.ncaust; });
        write_int([](const CurveRow& r) { return r.curve; });
        write_int([](const CurveRow& r) { return r.vertex; });
        write_int([](const CurveRow& r) { return r.closed; });
        write_dbl([](const CurveRow& r) { return r.ximg; });
        write_dbl([](const CurveRow& r) { return r.yimg; });
        write_dbl([](const CurveRow& r) { return r.r; });
        write_dbl([](const CurveRow& r) { return r.phi; });
        write_dbl([](const CurveRow& r) { return r.X; });
        write_dbl([](const CurveRow& r) { return r.Y; });
        write_dbl([](const CurveRow& r) { return r.g; });
        write_dbl([](const CurveRow& r) { return r.energy; });
        write_dbl([](const CurveRow& r) { return r.tdo; });
        write_dbl([](const CurveRow& r) { return r.tsd; });
        write_dbl([](const CurveRow& r) { return r.time; });
        write_int([](const CurveRow& r) { return r.indisc; });
        write_int([](const CurveRow& r) { return r.refined; });
        write_int([](const CurveRow& r) { return r.jslope; });
        write_int([](const CurveRow& r) { return r.accept; });
        write_dbl([](const CurveRow& r) { return r.jump; });
        for (int c = 0; c < ncols; c++) { free(c_names[c]); free(c_formats[c]); free(c_units[c]); }
    }

    // per-pixel image extensions
    auto write_axis_keywords = [&]()
    {
        fits.write_keyword("AXIS1", "Quantity along X axis", "Image plane X (rg)");
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
    Array2D<double> map(img_Nx, img_Ny);
    auto write_map = [&](const string& extname, const string& comment, const std::function<double(int)>& get)
    {
        for (int pix = 0; pix < nPix; pix++) map[pix / img_Ny][pix % img_Ny] = get(pix);
        fits.write_image(map, img_Nx, img_Ny, false);
        char name[16]; strncpy(name, extname.c_str(), 15); name[15] = 0;
        fits.set_ext_name(name);
        fits.write_comment(comment.c_str());
        write_axis_keywords();
    };
    for (int n = 1; n <= max_eqcross; n++)
    {
        const vector<SheetPixel>& S = sheets[n - 1];
        const string sn = to_string(n);
        write_map("JDISC" + sn, "Ray-family Jacobian at equatorial crossing " + sn + " (NaN: none)", [&](int p) { return S[p].valid ? S[p].J : NaN; });
        write_map("RDISC" + sn, "Boyer-Lindquist r at equatorial crossing " + sn, [&](int p) { return S[p].valid ? S[p].dc.r : NaN; });
        write_map("PHIDISC" + sn, "Boyer-Lindquist phi at equatorial crossing " + sn + " (rad)", [&](int p) { return S[p].valid ? S[p].dc.phi : NaN; });
        write_map("GDISC" + sn, "E_disc/E_obs for emission at equatorial crossing " + sn + " (NaN inside ISCO)", [&](int p) { return S[p].valid ? S[p].g : NaN; });
        write_map("TDISC" + sn, "Disc-to-observer coordinate time at equatorial crossing " + sn + " (rg/c)", [&](int p) { return S[p].valid ? S[p].dc.t : NaN; });
        write_map("NCDISC" + sn, "Caustic crossings along the ray before equatorial crossing " + sn, [&](int p) { return S[p].valid ? static_cast<double>(S[p].dc.ncaust) : NaN; });
    }
    write_map("NEQCROSS", "Equatorial plane crossings of the ray (up to MAXEQCR)", [&](int p) { return static_cast<double>(results[p].eqcross); });
    write_map("STATUS", "0 escaped, 1 horizon, 2 step limit, 3 bundle split, 4 skipped, 5 numerical breakdown, 6 stopped after MAXEQCR crossings",
              [&](int p) { return static_cast<double>(results[p].status); });

    // POINTS table: every traced point of the refined plane
    if (refine_levels > 0)
    {
        const long prows = static_cast<long>(results.size());
        const int ncols = 5;
        const char* names[ncols]   = { "XIMG", "YIMG", "LEVEL", "NEQCROSS", "STATUS" };
        const char* formats[ncols] = { "1D", "1D", "1J", "1J", "1J" };
        const char* units[ncols]   = { "rg", "rg", "", "", "" };
        char* c_names[ncols]; char* c_formats[ncols]; char* c_units[ncols];
        for (int c = 0; c < ncols; c++) { c_names[c] = strdup(names[c]); c_formats[c] = strdup(formats[c]); c_units[c] = strdup(units[c]); }
        char extname[] = "POINTS";
        fits.create_table(extname, ncols, prows, c_names, c_formats, c_units);
        fits.write_comment("All image-plane points traced: regular grid (LEVEL 0) and adaptive refinement");
        vector<double> dcol(prows); vector<int> icol(prows);
        for (long i = 0; i < prows; i++) dcol[i] = pts[i].x;
        fits.write_table_column(dcol.data(), prows);
        for (long i = 0; i < prows; i++) dcol[i] = pts[i].y;
        fits.write_table_column(dcol.data(), prows);
        for (long i = 0; i < prows; i++) icol[i] = pts[i].level;
        fits.write_table_column(icol.data(), prows);
        for (long i = 0; i < prows; i++) icol[i] = results[i].eqcross;
        fits.write_table_column(icol.data(), prows);
        for (long i = 0; i < prows; i++) icol[i] = results[i].status;
        fits.write_table_column(icol.data(), prows);
        for (int c = 0; c < ncols; c++) { free(c_names[c]); free(c_formats[c]); free(c_units[c]); }
    }

    // DISC table
    {
        const int ncols = 3;
        const char* names[ncols] = { "r", "time", "count" };
        const char* formats[ncols] = { "1D", "1D", "1J" };
        const char* units[ncols] = { "rg", "rg/c", "" };
        char* c_names[ncols]; char* c_formats[ncols]; char* c_units[ncols];
        for (int c = 0; c < ncols; c++) { c_names[c] = strdup(names[c]); c_formats[c] = strdup(formats[c]); c_units[c] = strdup(units[c]); }
        char extname[] = "DISC";
        fits.create_table(extname, ncols, stt.Nr, c_names, c_formats, c_units);
        fits.write_comment("Source-to-disc arrival time per annulus (mean over rays; NaN if none)");
        vector<double> rcol(stt.r), tcol(stt.t); vector<int> ccol(stt.count);
        fits.write_table_column(rcol.data(), stt.Nr);
        fits.write_table_column(tcol.data(), stt.Nr);
        fits.write_table_column(ccol.data(), stt.Nr);
        for (int c = 0; c < ncols; c++) { free(c_names[c]); free(c_formats[c]); free(c_units[c]); }
    }

    fits.close();
    for (int i = 0; i < 5; i++) delete planes[i];

    cout << "Done. Output: " << out_filename << endl;
    return 0;
}
