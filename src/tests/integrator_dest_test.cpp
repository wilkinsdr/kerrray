/*
 * integrator_dest_test.cpp
 *
 * Exercises the RayDestination overload of the symplectic integrator with an ImagePlane
 * source (backward tracing, negative spin) and a DiscWithISCODestination, i.e. the setup
 * used by imageplane_disc_image_isco and caustic_imageplane.
 *
 * Checks
 *  (a) every ray flagged RAY_STATUS_DEST sits on the disc surface (|theta - pi/2| < 1e-9)
 *      with r_isco <= r <= r_out -- the bisection landing on dest->reached()
 *  (b) the theta-limit overload and the destination overload give identical results for
 *      rays that land outside the ISCO (they are the same integration up to the stopping test)
 *  (c) agreement with the RK45 destination overload (tol 1e-10): termination class, and
 *      percentiles of |dr|, |dg| at the disc for rays landing in both
 *  (d) the total step count and wall time are reported for both methods
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
using namespace std;

#include "raytracer/imageplane.h"
#include "raytracer/ray_destination.h"
#include "include/kerr.h"

static const double SPIN  = 0.998;
static const double INCL  = 60.0 * M_PI / 180.0;
static const double DIST  = 1000.0;
// grid chosen so that no ray sits exactly at x = y = 0: ImagePlane produces NaN constants of
// motion for that ray, and the RK45 adaptive loop never terminates on a NaN state (the symplectic
// integrator flags such a ray as failed instead)
static const double X0 = -12.25, XMAX = 12.25, Y0 = -12.25, YMAX = 12.25, DX = 0.5;
static const double R_DISC = 500.0;

static int n_fail = 0;
static void check(bool ok, const string& what)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << endl;
    if (!ok) ++n_fail;
}
static double pct(vector<double> v, double p)
{
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    return v[min(v.size() - 1, (size_t)(p * v.size()))];
}

static ImagePlane<double>* make_plane()
{
    return new ImagePlane<double>(DIST, INCL, X0, XMAX, DX, Y0, YMAX, DX, SPIN, 0.0, PRECISION);
}

int main()
{
    const double r_isco = kerr_isco<double>(SPIN, +1);
    DiscWithISCODestination<double> dest(r_isco, R_DISC);
    cout << "ImagePlane: spin " << SPIN << ", incl 60 deg, " << (int)((XMAX - X0)/DX + 1) << "^2 rays, ISCO at " << r_isco << endl;

    // RK45 reference with destination
    ImagePlane<double>* ref = make_plane();
    ref->set_rk45_tol(1e-10);
    ref->redshift_start();
    auto t0 = chrono::high_resolution_clock::now();
    ref->run_raytrace(&dest, Integrator::RK45, 1.1 * DIST, 0);
    auto t1 = chrono::high_resolution_clock::now();
    ref->redshift(-1.0, true);
    const double t_ref = chrono::duration<double>(t1 - t0).count();

    // symplectic with destination
    ImagePlane<double>* sym = make_plane();
    sym->set_max_tstep(0.25);
    sym->redshift_start();
    t0 = chrono::high_resolution_clock::now();
    sym->run_raytrace(&dest, Integrator::Symplectic, 1.1 * DIST, 0);
    t1 = chrono::high_resolution_clock::now();
    sym->redshift(-1.0, true);
    const double t_sym = chrono::duration<double>(t1 - t0).count();

    // symplectic with the plain theta limit (for check b)
    ImagePlane<double>* symt = make_plane();
    symt->set_max_tstep(0.25);
    symt->redshift_start();
    symt->run_raytrace(Integrator::Symplectic, M_PI_2, 1.1 * DIST, 0);
    symt->redshift(-1.0, true);

    const int n = ref->get_count();
    int n_ref_steplim = 0, n_ref_steplim_horizon = 0, n_classified = 0, n_dest = 0, n_on_surface = 0, n_in_annulus = 0, n_valid = 0, n_agree = 0, n_both = 0, n_thetalim_same = 0, n_thetalim_cmp = 0;
    long steps_ref = 0, steps_sym = 0;
    double max_dtheta = 0, max_thetalim_diff = 0;
    vector<double> dr, dg, dt;
    for (int i = 0; i < n; i++)
    {
        const Ray<double>& a = ref->rays[i];
        const Ray<double>& b = sym->rays[i];
        const Ray<double>& c = symt->rays[i];
        if (a.steps == -1 && b.steps == -1) continue;
        ++n_valid;
        steps_ref += abs(a.steps); steps_sym += abs(b.steps);
        const bool da = a.status & RAY_STATUS_DEST, db = b.status & RAY_STATUS_DEST;
        if (db)
        {
            ++n_dest;
            max_dtheta = max(max_dtheta, abs(b.theta - M_PI_2));
            if (abs(b.theta - M_PI_2) < 1e-9) ++n_on_surface;
            if (b.r >= r_isco && b.r <= R_DISC) ++n_in_annulus;
            // (b) the theta-limit overload stops at the first equatorial crossing; if that crossing is
            //     outside the ISCO the two overloads must agree exactly
            if ((c.status & RAY_STATUS_DEST) && c.r >= r_isco && c.r <= R_DISC)
            {
                ++n_thetalim_cmp;
                const double d = max(abs(b.r - c.r), abs(b.t - c.t));
                max_thetalim_diff = max(max_thetalim_diff, d);
                if (d < 1e-10) ++n_thetalim_same;
            }
        }
        const bool ha = a.status & RAY_STATUS_HORIZON, hb = b.status & RAY_STATUS_HORIZON;
        const bool sa = a.status & RAY_STATUS_STEPLIM;
        if (sa) { ++n_ref_steplim; if (hb) ++n_ref_steplim_horizon; continue; }   // RK45 stalls on plunging rays
        ++n_classified;
        if ((da == db) && (ha == hb)) ++n_agree;
        if (da && db)
        {
            ++n_both;
            dr.push_back(abs(a.r - b.r));
            dg.push_back(abs(a.redshift - b.redshift));
            dt.push_back(abs(a.t - b.t));
        }
    }

    cout << fixed << setprecision(3);
    cout << "RK45 (tol 1e-10, dest): " << t_ref << " s, mean steps " << (double)steps_ref / n_valid << endl;
    cout << "Symplectic (default order " << sym->get_symplectic_order() << ", max_tstep 0.25, dest): " << t_sym << " s, mean steps " << (double)steps_sym / n_valid << endl;
    cout << scientific << setprecision(2);
    cout << "disc rays (symplectic): " << n_dest << ", max |theta - pi/2| = " << max_dtheta << endl;
    check(n_on_surface == n_dest, "(a) all destination rays land on the disc surface to 1e-9");
    check(n_in_annulus == n_dest, "(a) all destination rays lie within [r_isco, r_out]");
    cout << "theta-limit vs destination overload: " << n_thetalim_same << " / " << n_thetalim_cmp << " identical, max diff " << max_thetalim_diff << endl;
    check(n_thetalim_same == n_thetalim_cmp, "(b) theta-limit and destination overloads agree for rays landing outside the ISCO");
    cout << "termination agreement with RK45: " << n_agree << " / " << n_classified << " (" << fixed << setprecision(2) << 100.0 * n_agree / n_classified << "%)"
         << ";  rays RK45 could not classify (step limit): " << n_ref_steplim << ", of which " << n_ref_steplim_horizon << " go through the horizon here" << endl;
    cout << scientific << setprecision(2);
    cout << "  |dr| at disc: median " << pct(dr, 0.5) << "  90% " << pct(dr, 0.9) << "  max " << pct(dr, 1.0) << "  (" << n_both << " rays)" << endl;
    cout << "  |dt| at disc: median " << pct(dt, 0.5) << "  90% " << pct(dt, 0.9) << "  max " << pct(dt, 1.0) << endl;
    cout << "  |dg| at disc: median " << pct(dg, 0.5) << "  90% " << pct(dg, 0.9) << "  max " << pct(dg, 1.0) << endl;
    check(100.0 * n_agree / n_classified >= 99.0, "(c) termination agreement with RK45 >= 99%");
    check(pct(dr, 0.5) < 1e-4 && pct(dg, 0.5) < 1e-5, "(c) median |dr| < 1e-4 and median |dg| < 1e-5 at the disc");

    cout << endl << (n_fail == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << " (" << n_fail << " failures)" << endl;
    delete ref; delete sym; delete symt;
    return n_fail == 0 ? 0 : 1;
}
