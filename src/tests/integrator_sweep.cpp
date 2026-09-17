/*
 * integrator_sweep.cpp
 *
 * Work-precision benchmark of the four integrators (T3 of docs/plan_symplectic_integrator.md).
 *
 * For a fixed set of rays each method is run over a sweep of its accuracy knob:
 *   Euler / RK4   : precision  (50 ... 1600)
 *   RK45          : rk45_tol   (1e-6 ... 1e-11)
 *   Symplectic    : order (4, 6) x h0 (0.02 ... 0.0025) x max_tstep (1, 0.25, 0.1)
 *                   x far-field policy (constant cap, or cap growing with r beyond maxtstep_rlim)
 * and compared with a reference solution: symplectic order 6, h0 = 0.001, max_tstep = 0.05,
 * constant cap.  (RK45 at 1e-12 is not usable as the reference: it mis-traces the near-axis
 * class of lamppost rays described in integrator_compare_test.cpp and stalls on plunging rays.)
 *
 * Two ray sets, selected by the first argument:
 *   lamppost  : PointSource at r = 5, theta = 1e-3, spin 0.998, grid 0.05 x 0.05  (5040 rays)
 *   imageplane: ImagePlane at dist = 1000, incl = 60 deg, spin 0.998, |x|,|y| <= 12.25, dx = 0.5 (2500 rays)
 * Rays reaching the disc (theta = pi/2) in both the run and the reference are compared; for the
 * lamppost set the near-axis defect class is excluded so that the Carter-equation methods are
 * not penalised for it.
 *
 * Output: one CSV row per run appended to dat/integrator_sweep_<set>.csv with
 *   method, order, param1, param2, param3, wall_s, mean_steps, n_disc, n_diverged,
 *   dr_med, dr_p90, dg_med, dg_p90, dt_med, dt_p90, agree_frac
 * (dr, dg, dt: |difference| at the disc in radius, redshift, arrival time; diverged: |dr|/r > 1%;
 *  agree_frac: fraction of rays with the same termination class as the reference).
 * Plot with python/integrator_sweep_plot.py.
 *
 * Wall times use all available OpenMP threads; set OMP_NUM_THREADS=1 for single-thread numbers.
 *
 * The sweep can be run in parts (second argument): "ref" computes and caches the reference in
 * dat/integrator_sweep_<set>_ref.bin; "euler", "rk45", "symp4", "symp6" run the corresponding
 * sweeps against the cached reference and append to the CSV; "all" (default) does everything in
 * one process.  "quick" runs a reduced sweep for a smoke test.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstdlib>
using namespace std;

#include "raytracer/pointsource.h"
#include "raytracer/imageplane.h"
#include "include/kerr.h"

static const double SPIN = 0.998;

struct Setup
{
    string name;
    bool lamppost;
};

static Raytracer<double>* make_source(const Setup& s, double precision)
{
    if (s.lamppost)
    {
        double pos[4] = {0.0, 5.0, 1e-3, 0.0};
        PointSource<double>* src = new PointSource<double>(pos, 0.0, SPIN, precision, 0.05, 0.05);
        src->redshift_start();
        return src;
    }
    ImagePlane<double>* src = new ImagePlane<double>(1000.0, 60.0 * M_PI / 180.0, -12.25, 12.25, 0.5, -12.25, 12.25, 0.5, SPIN, 0.0, precision);
    src->redshift_start();
    return src;
}

struct RunResult
{
    Raytracer<double>* src;
    double seconds;
    double mean_steps;
};

// minimal per-ray record of the reference solution, cached between partial runs
struct RefRay { double r, t, redshift, alpha, beta; int steps, status; };

static vector<RefRay> ref_from_source(Raytracer<double>* src)
{
    vector<RefRay> v(src->get_count());
    for (int i = 0; i < src->get_count(); i++)
    {
        const Ray<double>& r = src->rays[i];
        v[i] = { r.r, r.t, r.redshift, r.alpha, r.beta, r.steps, r.status };
    }
    return v;
}
static void save_ref(const string& path, const vector<RefRay>& v)
{
    ofstream f(path, ios::binary); size_t n = v.size();
    f.write((const char*)&n, sizeof(n)); f.write((const char*)v.data(), n * sizeof(RefRay));
}
static vector<RefRay> load_ref(const string& path)
{
    ifstream f(path, ios::binary); size_t n = 0;
    if (!f) { cerr << "reference cache " << path << " not found: run with 'ref' first" << endl; exit(1); }
    f.read((char*)&n, sizeof(n)); vector<RefRay> v(n); f.read((char*)v.data(), n * sizeof(RefRay));
    return v;
}

static RunResult run(const Setup& s, Integrator method, double precision, double rk45_tol,
                     int order, double h0, double max_tstep, double tstep_rlim)
{
    Raytracer<double>* src = make_source(s, precision);
    src->set_rk45_tol(rk45_tol);
    src->set_symplectic_order(order);
    src->set_symplectic_step(h0);
    if (method == Integrator::Symplectic) src->set_max_tstep(max_tstep, tstep_rlim);

    auto t0 = chrono::high_resolution_clock::now();
    src->run_raytrace(method, M_PI_2, s.lamppost ? 1000.0 : 1100.0, 0);   // ImagePlane rays start at dist = 1000
    auto t1 = chrono::high_resolution_clock::now();
    src->redshift(-1.0, !s.lamppost);   // reverse = true for the ImagePlane backward trace

    long steps = 0; int n = 0;
    for (int i = 0; i < src->get_count(); i++)
        if (src->rays[i].steps != -1) { steps += abs(src->rays[i].steps); ++n; }
    return { src, chrono::duration<double>(t1 - t0).count(), (double)steps / max(n, 1) };
}

static int term_class(int status)
{
    if (status & RAY_STATUS_STEPLIM) return 3;
    if (status & RAY_STATUS_HORIZON) return 2;
    if (status & RAY_STATUS_RLIM)    return 1;
    if (status & RAY_STATUS_DEST)    return 0;
    return 4;
}

static double pct(vector<double> v, double p)
{
    if (v.empty()) return NAN;
    sort(v.begin(), v.end());
    return v[min(v.size() - 1, (size_t)(p * v.size()))];
}

static void compare_and_log(ofstream& csv, const Setup& s, const string& method, int order,
                            double p1, double p2, double p3, const RunResult& run, const vector<RefRay>& ref)
{
    int n_disc = 0, n_div = 0, n_agree = 0, n_valid = 0;
    vector<double> dr, dg, dt;
    for (size_t i = 0; i < ref.size(); i++)
    {
        const Ray<double>& a = run.src->rays[i];
        const RefRay& b = ref[i];
        if (a.steps == -1 || b.steps == -1) continue;
        if (s.lamppost && abs(a.alpha) <= 0.15 && sin(a.beta) > 0) continue;   // Carter-propagator defect class
        ++n_valid;
        if (term_class(a.status) == term_class(b.status)) ++n_agree;
        if (term_class(a.status) == 0 && term_class(b.status) == 0)
        {
            ++n_disc;
            if (abs(a.r - b.r) / b.r > 1e-2) { ++n_div; continue; }
            dr.push_back(abs(a.r - b.r));
            dg.push_back(abs(a.redshift - b.redshift));
            dt.push_back(abs(a.t - b.t));
        }
    }
    csv << method << "," << order << "," << p1 << "," << p2 << "," << p3 << ","
        << run.seconds << "," << run.mean_steps << "," << n_disc << "," << n_div << ","
        << pct(dr, 0.5) << "," << pct(dr, 0.9) << "," << pct(dg, 0.5) << "," << pct(dg, 0.9) << ","
        << pct(dt, 0.5) << "," << pct(dt, 0.9) << "," << (double)n_agree / max(n_valid, 1) << endl;
    cout << setw(10) << method << " o" << order << " p=(" << p1 << "," << p2 << "," << p3 << ")  "
         << fixed << setprecision(3) << run.seconds << " s  steps " << setprecision(0) << run.mean_steps
         << scientific << setprecision(2) << "  dr50 " << pct(dr, 0.5) << " dr90 " << pct(dr, 0.9)
         << "  dg50 " << pct(dg, 0.5) << " dg90 " << pct(dg, 0.9) << "  div " << n_div << "/" << n_disc << endl;
}

int main(int argc, char** argv)
{
    Setup s;
    s.name = (argc > 1) ? argv[1] : "lamppost";
    s.lamppost = (s.name == "lamppost");
    const string part = (argc > 2) ? argv[2] : "all";
    const bool quick = (part == "quick");
    const bool all = (part == "all" || quick);
    const string ref_path = "dat/integrator_sweep_" + s.name + "_ref.bin";
    const string csv_path = "dat/integrator_sweep_" + s.name + ".csv";

    cout << "=== integrator work-precision sweep: " << s.name << " (" << part << ") ===" << endl;

    // --- reference ---
    vector<RefRay> ref;
    if (all || part == "ref")
    {
        cout << "reference: symplectic order 6, h0 = 0.001, max_tstep = 0.05, constant cap" << endl;
        RunResult rr = run(s, Integrator::Symplectic, PRECISION, 1e-8, 6, 0.001, 0.05, 1e12);
        cout << "  " << rr.seconds << " s, mean steps " << rr.mean_steps << endl;
        ref = ref_from_source(rr.src); delete rr.src;
        save_ref(ref_path, ref);
        ofstream csv(csv_path);
        csv << "method,order,p1,p2,p3,wall_s,mean_steps,n_disc,n_diverged,dr_med,dr_p90,dg_med,dg_p90,dt_med,dt_p90,agree_frac" << endl;
        if (part == "ref") return 0;
    }
    else
        ref = load_ref(ref_path);

    ofstream csv(csv_path, ios::app);
    csv << scientific << setprecision(6);

    const vector<double> precisions = quick ? vector<double>{100, 400} : vector<double>{50, 100, 200, 400, 800, 1600};
    if (all || part == "euler")
        for (double p : precisions)
        {
            RunResult e = run(s, Integrator::Euler, p, 1e-8, 4, -1, MAXDT, MAXDT_RLIM);
            compare_and_log(csv, s, "euler", 1, p, 0, 0, e, ref); delete e.src;
            RunResult r4 = run(s, Integrator::RK4, p, 1e-8, 4, -1, MAXDT, MAXDT_RLIM);
            compare_and_log(csv, s, "rk4", 4, p, 0, 0, r4, ref); delete r4.src;
        }
    const vector<double> tols = quick ? vector<double>{1e-8} : vector<double>{1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11};
    if (all || part == "rk45")
        for (double tol : tols)
        {
            RunResult r = run(s, Integrator::RK45, PRECISION, tol, 4, -1, MAXDT, MAXDT_RLIM);
            compare_and_log(csv, s, "rk45", 5, tol, 0, 0, r, ref); delete r.src;
        }
    const vector<double> h0s = quick ? vector<double>{0.01} : vector<double>{0.02, 0.01, 0.005, 0.0025};
    const vector<double> mdts = quick ? vector<double>{0.25} : vector<double>{1.0, 0.25, 0.1};
    for (int order : {4, 6})
    {
        if (!(all || part == (order == 4 ? "symp4" : "symp6"))) continue;
        for (double h0 : h0s)
            for (double mdt : mdts)
                for (double rlim : {1e12, 100.0})
                {
                    RunResult r = run(s, Integrator::Symplectic, PRECISION, 1e-8, order, h0, mdt, rlim);
                    compare_and_log(csv, s, (rlim < 1e6) ? "symp_grow" : "symp", order, h0, mdt, rlim, r, ref); delete r.src;
                }
    }
    cout << "appended to " << csv_path << endl;
    return 0;
}
