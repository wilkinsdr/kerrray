//
// caustic_3d.cpp
//
// 3D caustic mapping for an observer in the Kerr spacetime.
//
// Maps the caustic surface of the congruence of null geodesics reaching a distant observer at
// Boyer-Lindquist (r = dist, theta = incl): the locus of points where neighbouring rays of the family
// cross (Rauch & Blandford 1994; Bozza 2008).  Rays are traced backwards from the observer's image plane
// (ImagePlane initialiser, so the family is parametrised by the image-plane coordinates x, y) until they
// fall through the horizon or escape past r_max, passing through the equatorial plane.
//
// Method: for every image-plane pixel a bundle of five rays is traced in lockstep with the Mino-time
// symplectic stepper (mino_stepper.h): the centre ray at (x, y) and four offset rays at (x +/- delta, y),
// (x, y +/- delta).  At every step the Jacobian of the ray family
//
//     J(tau) = det[ dX/dx , dX/dy , dX/dtau ]
//
// is formed from central differences of the Cartesian positions of the offset rays and the Mino-time
// velocity of the centre ray.  J vanishes on the caustic surface; each sign change of J along the centre
// ray is located to high precision by bisection on the fraction of the step (re-taking the composed step
// from the saved previous states of all five rays) and recorded as a caustic point.
//
// Only the sign of J matters for locating the caustics, so the finite-difference bundle can be kept in
// the linear regime near the photon shell (where separations grow by ~e^6 per orbit) by rescaling each
// offset pair's deviation from the centre ray -- in the full state (u, p_u, theta, p_theta, t, phi) and the
// constant of motion h -- back to delta whenever it exceeds delta_max.  A positive rescaling does not
// change sign(J); the accumulated log10 scale factor is stored with each caustic point as a diagnostic.
//
// FITS output:
//   primary header  -- run parameters
//   CAUSTICS        -- binary table, one row per caustic crossing: IX, IY (pixel), XIMG, YIMG (image-plane
//                      coordinates, rg), N (crossing index along the ray, 1-based), TAU (Mino time from the
//                      image plane), T, R, THETA, PHI (Boyer-Lindquist; PHI wrapped to (-pi, pi]), X, Y, Z
//                      (Cartesian, kerr.h cartesian()), RFLIPS (radial turning points so far), EQCROSS
//                      (equatorial crossings so far), DJSIGN (+1: J went - to +, -1: + to -), LOGSCALE
//                      (accumulated log10 bundle rescaling)
//   NCAUST          -- image: number of caustic crossings along the ray
//   STATUS          -- image: 0 escaped (r >= r_max), 1 horizon, 2 step limit, 3 bundle split (an offset ray
//                      fell through the horizon before the centre ray), 4 skipped (undefined initial ray),
//                      5 numerical breakdown
//   R1, THETA1, PHI1 -- image: Boyer-Lindquist position of the first caustic crossing (0 if none)
//   EQCROSS         -- image: total equatorial crossings of the ray
//   TAU_END         -- image: Mino time at which the ray terminated
//
// Array dimensions: the user specifies Nx, Ny grid steps; the image plane has (Nx+1)*(Ny+1) rays
// (fencepost convention) and all image extensions have dimensions Nx+1, Ny+1.
//
// Observer frame (for plotting): the image-plane centre lies in the direction
//     n = (sin i cos phi0, sin i sin phi0, cos i)
// of the Cartesian frame; the image-plane x axis is along +phi (-sin phi0, cos phi0, 0) and the y axis
// points towards the pole (-cos i cos phi0, -cos i sin phi0, sin i).
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdio>

using namespace std;

#include "../raytracer/imageplane.h"
#include "caustic_bundle.h"

#include "../include/fits_output.h"
#include "../include/array.h"
#include "../include/par_file.h"
#include "../include/par_args.h"
#include "../include/kerr.h"
#include "../include/progress_bar.h"

int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile")
                          ? par_args.get_string_parameter("--parfile")
                          : "../par/caustic_3d.par";

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

    double x0   = par_file.get_parameter<double>("x0");
    double xmax = par_file.get_parameter<double>("xmax");
    int    Nx   = par_args.key_exists("--Nx")
                  ? par_args.get_parameter<int>("--Nx")
                  : par_file.get_parameter<int>("Nx");
    double y0   = par_file.get_parameter<double>("y0",   x0);
    double ymax = par_file.get_parameter<double>("ymax", xmax);
    int    Ny   = par_args.key_exists("--Ny")
                  ? par_args.get_parameter<int>("--Ny")
                  : par_file.get_parameter<int>("Ny", Nx);

    double delta     = par_args.key_exists("--delta")
                       ? par_args.get_parameter<double>("--delta")
                       : par_file.get_parameter<double>("delta", 1e-4);      // bundle offset (rg)
    double delta_max = par_file.get_parameter<double>("delta_max", 100*delta); // rescale threshold (rg)
    double r_max     = par_file.get_parameter<double>("r_max", 1.1 * dist);   // escape radius (rg)

    double symp_step   = par_file.get_parameter<double>("symp_step", -1);     // Mino-time step (<= 0: 1/precision)
    int    symp_order  = par_file.get_parameter<int>("symp_order", 6);        // 2, 4 or 6
    double max_tstep   = par_file.get_parameter<double>("max_tstep", MAXDT);
    double maxtstep_rlim = par_file.get_parameter<double>("maxtstep_rlim", MAXDT_RLIM);
    double max_phistep = par_file.get_parameter<double>("max_phistep", MAXDPHI);
    double precision   = par_file.get_parameter<double>("precision", PRECISION);
    int    steplim     = par_file.get_parameter<int>("steplim", SYMP_STEPLIM);
    int    max_caustics = par_file.get_parameter<int>("max_caustics", 32);
    // diagnostics: dump the centre ray and Jacobian of one bundle to a text file (dump_pixel = "ix,iy")
    string dump_pixel = par_args.key_exists("--dump_pixel")
                        ? par_args.get_parameter<string>("--dump_pixel")
                        : par_file.get_parameter<string>("dump_pixel", "");
    string dump_file  = par_args.key_exists("--dump_file")
                        ? par_args.get_parameter<string>("--dump_file")
                        : par_file.get_parameter<string>("dump_file", "../dat/caustic_3d_ray.dat");
    int show_progress = par_args.key_exists("--show_progress")
                        ? par_args.get_parameter<int>("--show_progress")
                        : par_file.get_parameter<int>("show_progress", 1);

    // Step sizes for the ray grid.  ImagePlane computes its ray count as (xmax-x0)/dx + 1 (fencepost, truncated),
    // so the step is shrunk by a relative 1e-9 to guarantee exactly Nx+1 rays per axis for every plane.
    const double dx = (xmax - x0) / Nx;
    const double dy = (ymax - y0) / Ny;
    const double dx_ip = dx * (1 - 1e-9);
    const double dy_ip = dy * (1 - 1e-9);
    const int img_Nx = Nx + 1;
    const int img_Ny = Ny + 1;
    const int nPix = img_Nx * img_Ny;

    const double r_horizon = kerr_horizon<double>(spin);
    const double r_isco = kerr_isco<double>(spin, +1);

    cout << "*****" << endl;
    cout << "Observer at r = " << dist << ", incl = " << incl << " deg, phi0 = " << plane_phi0 << endl;
    cout << "Spin a = " << spin << "  (horizon " << r_horizon << ", ISCO " << r_isco << ")" << endl;
    cout << "Image plane: " << img_Nx << " x " << img_Ny << " = " << nPix << " bundles of 5 rays" << endl;
    cout << "Bundle offset delta = " << delta << ", rescale beyond " << delta_max << endl;
    cout << "*****" << endl << endl;

    // --- ray initialisation: five image planes, offset by +/-delta in x and y ---
    // b[0] centre, b[1] +x, b[2] -x, b[3] +y, b[4] -y
    const double offx[5] = { 0,  delta, -delta, 0, 0 };
    const double offy[5] = { 0, 0, 0,  delta, -delta };
    ImagePlane<double>* planes[5];
    for (int i = 0; i < 5; i++)
    {
        planes[i] = new ImagePlane<double>(dist, incl, x0 + offx[i], xmax + offx[i], dx_ip,
                                           y0 + offy[i], ymax + offy[i], dy_ip, spin, plane_phi0, precision);
        if (planes[i]->get_count() != nPix)
        {
            cerr << "Error: image plane " << i << " has " << planes[i]->get_count()
                 << " rays, expected " << nPix << endl;
            return 1;
        }
    }

    BundleParams P;
    P.a = -spin;               // ImagePlane traces backwards in time by negating the spin
    P.horizon = r_horizon;
    P.h0 = (symp_step > 0) ? symp_step : 1.0 / precision;
    P.r_cap = 2 * r_horizon;
    P.max_tstep = max_tstep;
    P.maxtstep_rlim = maxtstep_rlim;
    P.max_phistep = max_phistep;
    P.r_max = r_max;
    P.delta = delta;
    P.delta_max = delta_max;
    P.order = symp_order;
    P.steplim = steplim;
    P.max_caustics = max_caustics;

    const Ray<double>* ray_arrays[5];
    for (int i = 0; i < 5; i++) ray_arrays[i] = planes[i]->rays;

    // --- per-pixel results ---
    vector<PixelResult> results(nPix);
    vector<CausticPoint> caustics;

    cout << "Tracing ray bundles..." << endl;
    ProgressBar prog(nPix, "Bundle", 0, (show_progress > 0));
    show_progress = abs(show_progress);
    int pix_done = 0;

    #pragma omp parallel
    {
        vector<CausticPoint> local;

        #pragma omp for schedule(dynamic)
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

            const int ix = planes[0]->get_x_index(pix);
            const int iy = planes[0]->get_y_index(pix);
            const double ximg = planes[0]->rays[pix].alpha;
            const double yimg = planes[0]->rays[pix].beta;

            BundleRay b[5];
            const bool ok = bundle_from_rays(ray_arrays, pix, P.a, b);

            PixelResult& res = results[pix];
            if (!ok)
            {
                // e.g. the x = y = 0 ray, whose constants of motion are undefined in ImagePlane
                res.status = STATUS_SKIPPED;
                res.ncaust = 0; res.eqcross = 0; res.tau_end = 0;
                res.r1 = res.theta1 = res.phi1 = 0;
                continue;
            }

            trace_bundle(b, P, ix, iy, ximg, yimg, local, res);
        }

        #pragma omp critical
        caustics.insert(caustics.end(), local.begin(), local.end());
    }
    prog.done();

    // --- optional diagnostic dump of a single bundle (re-traced serially) ---
    if (dump_pixel.length() > 0)
    {
        int dix = -1, diy = -1;
        if (sscanf(dump_pixel.c_str(), "%d,%d", &dix, &diy) == 2 && dix >= 0 && dix < img_Nx && diy >= 0 && diy < img_Ny)
        {
            const int pix = dix * img_Ny + diy;
            BundleRay b[5];
            bundle_from_rays(ray_arrays, pix, P.a, b);
            ofstream dump(dump_file.c_str());
            dump << "# step tau t r theta phi X Y Z J logscale rflips eqcross   (pixel " << dix << "," << diy
                 << " x=" << planes[0]->rays[pix].alpha << " y=" << planes[0]->rays[pix].beta << ")\n";
            vector<CausticPoint> dummy;
            PixelResult dres;
            trace_bundle(b, P, dix, diy, planes[0]->rays[pix].alpha, planes[0]->rays[pix].beta, dummy, dres, &dump);
            cout << "Wrote ray dump for pixel (" << dix << "," << diy << ") to " << dump_file << endl;
        }
        else
            cerr << "Warning: dump_pixel must be \"ix,iy\" within the image; ignored" << endl;
    }

    for (int i = 0; i < 5; i++) delete planes[i];

    sort(caustics.begin(), caustics.end(), [](const CausticPoint& p, const CausticPoint& q)
    {
        if (p.ix != q.ix) return p.ix < q.ix;
        if (p.iy != q.iy) return p.iy < q.iy;
        return p.n < q.n;
    });

    // --- summary ---
    long cnt_status[6] = {0, 0, 0, 0, 0, 0};
    long total_caust = 0, max_n = 0;
    for (int pix = 0; pix < nPix; pix++)
    {
        ++cnt_status[results[pix].status];
        total_caust += results[pix].ncaust;
        max_n = max<long>(max_n, results[pix].ncaust);
    }
    cout << "Caustic crossings: " << caustics.size() << " stored (" << total_caust << " found, max "
         << max_n << " per ray)" << endl;
    cout << "  rays: escaped=" << cnt_status[STATUS_ESCAPED] << " horizon=" << cnt_status[STATUS_HORIZON]
         << " steplim=" << cnt_status[STATUS_STEPLIM] << " split=" << cnt_status[STATUS_SPLIT]
         << " skipped=" << cnt_status[STATUS_SKIPPED] << " nan=" << cnt_status[STATUS_NAN] << endl;

    // --- FITS output ---
    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Kerr spacetime 3D caustic map (observer image-plane ray bundles)");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "caustic_3d");
    fits.write_keyword("DIST",    "Observer distance / image plane distance (rg)", dist);
    fits.write_keyword("INCL",    "Observer inclination (degrees from spin axis)", incl);
    fits.write_keyword("PHI0",    "Azimuth of the image plane centre (radians)", plane_phi0);
    fits.write_keyword("SPIN",    "Black hole spin parameter a/M", spin);
    fits.write_keyword("HORIZON", "Event horizon radius (rg)", r_horizon);
    fits.write_keyword("ISCO",    "Innermost stable circular orbit (rg)", r_isco);
    fits.write_keyword("DELTA",   "Bundle offset in the image plane (rg)", delta);
    fits.write_keyword("DELTAMAX", "Bundle rescaling threshold (rg)", delta_max);
    fits.write_keyword("RMAX",    "Escape radius (rg)", r_max);
    fits.write_keyword("SYMPSTEP", "Mino-time step in the strong field", P.h0);
    fits.write_keyword("SYMPORD", "Symplectic composition order", symp_order);
    fits.write_keyword("MAXTSTEP", "Far-field cap on the coordinate-time step", max_tstep);
    fits.write_keyword("NX",   "Number of pixels in X", img_Nx);
    fits.write_keyword("NY",   "Number of pixels in Y", img_Ny);
    fits.write_keyword("X0",   "Start of X axis (rg)", x0);
    fits.write_keyword("XMAX", "End of X axis (rg)", xmax);
    fits.write_keyword("Y0",   "Start of Y axis (rg)", y0);
    fits.write_keyword("YMAX", "End of Y axis (rg)", ymax);
    fits.write_keyword("NCAUSTS", "Number of caustic points stored", static_cast<long>(caustics.size()));

    // CAUSTICS table
    {
        const long rows = static_cast<long>(caustics.size());
        const int ncols = 17;
        const char* names[ncols]   = { "IX", "IY", "XIMG", "YIMG", "N", "TAU", "T", "R", "THETA", "PHI",
                                       "X", "Y", "Z", "RFLIPS", "EQCROSS", "DJSIGN", "LOGSCALE" };
        const char* formats[ncols] = { "1J", "1J", "1D", "1D", "1J", "1D", "1D", "1D", "1D", "1D",
                                       "1D", "1D", "1D", "1J", "1J", "1J", "1D" };
        const char* units[ncols]   = { "", "", "rg", "rg", "", "", "rg/c", "rg", "rad", "rad",
                                       "rg", "rg", "rg", "", "", "", "" };
        char* c_names[ncols]; char* c_formats[ncols]; char* c_units[ncols];
        for (int c = 0; c < ncols; c++)
        {
            c_names[c] = strdup(names[c]); c_formats[c] = strdup(formats[c]); c_units[c] = strdup(units[c]);
        }
        char extname[] = "CAUSTICS";
        fits.create_table(extname, ncols, rows, c_names, c_formats, c_units);
        fits.write_comment("One row per caustic crossing along a ray (sign change of the ray-family Jacobian)");
        fits.write_comment("DJSIGN: +1 if J went from - to +, -1 if from + to -; LOGSCALE: log10 bundle rescaling");

        vector<int> icol(rows);
        vector<double> dcol(rows);
        auto write_int = [&](int (*get)(const CausticPoint&))
        {
            for (long i = 0; i < rows; i++) icol[i] = get(caustics[i]);
            fits.write_table_column(icol.data(), rows);
        };
        auto write_dbl = [&](double (*get)(const CausticPoint&))
        {
            for (long i = 0; i < rows; i++) dcol[i] = get(caustics[i]);
            fits.write_table_column(dcol.data(), rows);
        };
        write_int([](const CausticPoint& p) { return p.ix; });
        write_int([](const CausticPoint& p) { return p.iy; });
        write_dbl([](const CausticPoint& p) { return p.ximg; });
        write_dbl([](const CausticPoint& p) { return p.yimg; });
        write_int([](const CausticPoint& p) { return p.n; });
        write_dbl([](const CausticPoint& p) { return p.tau; });
        write_dbl([](const CausticPoint& p) { return p.t; });
        write_dbl([](const CausticPoint& p) { return p.r; });
        write_dbl([](const CausticPoint& p) { return p.theta; });
        write_dbl([](const CausticPoint& p) { return p.phi; });
        write_dbl([](const CausticPoint& p) { return p.X; });
        write_dbl([](const CausticPoint& p) { return p.Y; });
        write_dbl([](const CausticPoint& p) { return p.Z; });
        write_int([](const CausticPoint& p) { return p.rflips; });
        write_int([](const CausticPoint& p) { return p.eqcross; });
        write_int([](const CausticPoint& p) { return p.dj_sign; });
        write_dbl([](const CausticPoint& p) { return p.logscale; });

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
    auto write_map = [&](const char* extname, const char* comment, double (*get)(const PixelResult&))
    {
        for (int pix = 0; pix < nPix; pix++)
            map[pix / img_Ny][pix % img_Ny] = get(results[pix]);
        fits.write_image(map, img_Nx, img_Ny, false);
        char name[16]; strncpy(name, extname, 15); name[15] = 0;
        fits.set_ext_name(name);
        fits.write_comment(comment);
        write_axis_keywords();
    };
    write_map("NCAUST",  "Number of caustic crossings along the ray",
              [](const PixelResult& r) { return static_cast<double>(r.ncaust); });
    write_map("STATUS",  "0 escaped, 1 horizon, 2 step limit, 3 bundle split, 4 skipped, 5 numerical breakdown",
              [](const PixelResult& r) { return static_cast<double>(r.status); });
    write_map("R1",      "Boyer-Lindquist r of the first caustic crossing (rg); 0 if none",
              [](const PixelResult& r) { return r.r1; });
    write_map("THETA1",  "Boyer-Lindquist theta of the first caustic crossing (rad); 0 if none",
              [](const PixelResult& r) { return r.theta1; });
    write_map("PHI1",    "Boyer-Lindquist phi of the first caustic crossing (rad); 0 if none",
              [](const PixelResult& r) { return r.phi1; });
    write_map("EQCROSS", "Total number of equatorial plane crossings of the ray",
              [](const PixelResult& r) { return static_cast<double>(r.eqcross); });
    write_map("TAU_END", "Mino time at which the ray terminated",
              [](const PixelResult& r) { return r.tau_end; });

    fits.close();

    cout << "Done. Output: " << out_filename << endl;
    return 0;
}
