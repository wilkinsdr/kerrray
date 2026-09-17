//
// caustic_3d_test.cpp
//
// Self-checking test of the lockstep ray-bundle caustic tracer (src/caustic/caustic_bundle.h) in the
// Schwarzschild limit, where the caustics of the family of rays reaching a distant observer are known
// exactly: they lie on the observer's optical axis (the line through the black hole along the line of
// sight), behind the hole for the primary caustic and alternately in front of / behind it for the
// higher-order ones (Rauch & Blandford 1994; Bozza 2008).
//
// Checks:
//   (a) every caustic crossing found for a = 0 lies on the optical axis to within the finite-distance
//       image-plane tolerance (perpendicular offset < 1e-2 rg at dist = 1000; the residual scales as 1/dist^2)
//   (b) the primary caustic (N = 1) lies behind the black hole (Z' < 0), at a distance that increases with
//       the impact parameter, and every escaping ray outside the shadow crosses it exactly once before
//       reaching r_max (rays that fall through the horizon may cross none)
//   (c) the ImagePlane initial data is consistent (rays are parallel): the Jacobian of the family does not
//       change sign between the image plane and r = 50 for a Schwarzschild ray with b = 6
//   (d) the result is insensitive to the bundle offset: delta = 1e-3 and 1e-5 give the same primary
//       caustic positions to < 1e-3 rg
//
// No cfitsio dependency:
//   g++ -O2 -std=c++17 -fopenmp -I src src/tests/caustic_3d_test.cpp src/raytracer/raytracer.cpp src/raytracer/imageplane.cpp
//
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>
using namespace std;

#include "../raytracer/imageplane.h"
#include "../caustic/caustic_bundle.h"

static int n_fail = 0;
static void check(bool ok, const char* what)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << endl;
    if (!ok) ++n_fail;
}

struct RunResult
{
    vector<CausticPoint> points;
    vector<PixelResult> pixels;
    int nPix;
};

static RunResult run_schwarzschild(double dist, double delta, double extent, int Ngrid)
{
    const double spin = 0, incl = 60, phi0 = 0;
    const double dx = 2*extent / Ngrid * (1 - 1e-9);
    const double offx[5] = { 0,  delta, -delta, 0, 0 };
    const double offy[5] = { 0, 0, 0,  delta, -delta };
    ImagePlane<double>* planes[5];
    for (int i = 0; i < 5; i++)
        planes[i] = new ImagePlane<double>(dist, incl, -extent + offx[i], extent + offx[i], dx,
                                           -extent + offy[i], extent + offy[i], dx, spin, phi0, PRECISION);
    const Ray<double>* ray_arrays[5];
    for (int i = 0; i < 5; i++) ray_arrays[i] = planes[i]->rays;

    BundleParams P;
    P.a = -spin; P.horizon = kerr_horizon<double>(spin); P.h0 = 0.01; P.r_cap = 2*P.horizon;
    P.max_tstep = MAXDT; P.maxtstep_rlim = MAXDT_RLIM; P.max_phistep = MAXDPHI;
    P.r_max = 1.1*dist; P.delta = delta; P.delta_max = 100*delta; P.order = 6; P.steplim = SYMP_STEPLIM;
    P.max_caustics = 32;

    RunResult R;
    R.nPix = planes[0]->get_count();
    R.pixels.resize(R.nPix);
    #pragma omp parallel
    {
        vector<CausticPoint> local;
        #pragma omp for schedule(dynamic)
        for (int pix = 0; pix < R.nPix; pix++)
        {
            BundleRay b[5];
            if (!bundle_from_rays(ray_arrays, pix, P.a, b)) { R.pixels[pix].status = STATUS_SKIPPED; R.pixels[pix].ncaust = 0; continue; }
            trace_bundle(b, P, planes[0]->get_x_index(pix), planes[0]->get_y_index(pix),
                         planes[0]->rays[pix].alpha, planes[0]->rays[pix].beta, local, R.pixels[pix]);
        }
        #pragma omp critical
        R.points.insert(R.points.end(), local.begin(), local.end());
    }
    for (int i = 0; i < 5; i++) delete planes[i];
    return R;
}

int main()
{
    cout << "caustic_3d_test: Schwarzschild caustics of a distant observer (incl = 60 deg)" << endl;

    const double dist = 1000;
    const double i = 60 * M_PI / 180;
    const double n[3] = { sin(i), 0, cos(i) };   // line of sight (phi0 = 0)

    RunResult R = run_schwarzschild(dist, 1e-4, 8.0, 40);
    cout << "  " << R.nPix << " bundles, " << R.points.size() << " caustic crossings" << endl;

    // (a) all crossings on the optical axis
    double perp_max = 0;
    int n1 = 0, n1_behind = 0;
    for (const CausticPoint& p : R.points)
    {
        const double along = p.X*n[0] + p.Y*n[1] + p.Z*n[2];
        const double px = p.X - along*n[0], py = p.Y - along*n[1], pz = p.Z - along*n[2];
        const double perp = sqrt(px*px + py*py + pz*pz);
        if (perp > perp_max) perp_max = perp;
        if (p.n == 1) { ++n1; if (along < 0) ++n1_behind; }
    }
    cout << "  max perpendicular offset from the optical axis: " << perp_max << endl;
    check(R.points.size() > 500, "(a) caustic crossings were found");
    check(perp_max < 1e-2, "(a) all caustic crossings lie on the optical axis (offset < 1e-2 rg)");

    // (b) primary caustic behind the hole, once per escaping ray, distance increasing with b
    check(n1 > 0 && n1_behind == n1, "(b) all primary (N = 1) crossings are behind the black hole");
    int escaped = 0, escaped_one = 0, horizon_none_or_more = 0, horizon = 0;
    for (const PixelResult& q : R.pixels)
    {
        if (q.status == STATUS_ESCAPED) { ++escaped; if (q.ncaust >= 1) ++escaped_one; }
        if (q.status == STATUS_HORIZON) ++horizon;
    }
    cout << "  escaped rays " << escaped << " (with >= 1 crossing: " << escaped_one << "), captured " << horizon << endl;
    check(escaped > 0 && escaped_one == escaped, "(b) every escaping ray crosses the primary caustic");
    // distance behind the hole increases with impact parameter (compare the b-ordered N = 1 crossings)
    map<double, double> dist_by_b;
    for (const CausticPoint& p : R.points)
        if (p.n == 1) dist_by_b[hypot(p.ximg, p.yimg)] = -(p.X*n[0] + p.Y*n[1] + p.Z*n[2]);
    int nonmono = 0; double prev_b = -1, prev_d = -1;
    for (auto& kv : dist_by_b)
    {
        if (prev_b > 0 && kv.first - prev_b > 0.05 && kv.second < prev_d - 1e-3) ++nonmono;
        prev_b = kv.first; prev_d = kv.second;
    }
    check(nonmono == 0, "(b) primary caustic distance increases monotonically with impact parameter");

    // (c) parallel initial data: J keeps its sign from the plane to r = 50 for b = 6 (ray at x = 6, y = 0)
    {
        const double delta = 1e-4;
        const double offx[5] = { 0,  delta, -delta, 0, 0 };
        const double offy[5] = { 0, 0, 0,  delta, -delta };
        ImagePlane<double>* planes[5];
        for (int k = 0; k < 5; k++)
            planes[k] = new ImagePlane<double>(dist, 60, 6 + offx[k], 6 + offx[k], 1, offy[k], offy[k], 1, 0, 0, PRECISION);
        const Ray<double>* ray_arrays[5];
        for (int k = 0; k < 5; k++) ray_arrays[k] = planes[k]->rays;
        BundleRay b[5];
        bundle_from_rays(ray_arrays, 0, 0.0, b);
        MinoStepper<double> st[5] = { MinoStepper<double>(0, b[0].k, b[0].h, 6), MinoStepper<double>(0, b[1].k, b[1].h, 6),
                                      MinoStepper<double>(0, b[2].k, b[2].h, 6), MinoStepper<double>(0, b[3].k, b[3].h, 6),
                                      MinoStepper<double>(0, b[4].k, b[4].h, 6) };
        const double J0 = bundle_jacobian(b, 0.0, delta);
        double Jmin_rel = 1;
        double r = dist;
        for (int step = 0; step < 100000 && r > 50; step++)
        {
            double tdot, phidot, sd;
            const double h = mino_step_size<double>(r, b[0].s.theta, b[0].k, b[0].h, 0.0, 0.01, 4, MAXDT, MAXDT_RLIM, MAXDPHI, tdot, phidot);
            bundle_step(b, st, h);
            mino_r_from_u<double>(b[0].s.u, 0.0, r, sd);
            const double J = bundle_jacobian(b, 0.0, delta) / (r*r);   // J scales as r^2 through dX/dtau
            Jmin_rel = min(Jmin_rel, J / (J0 / (dist*dist)));
        }
        cout << "  min J(r)/r^2 relative to the image plane between r = 1000 and 50: " << Jmin_rel << endl;
        check(Jmin_rel > 0.5, "(c) ImagePlane rays are parallel: J/r^2 stays within a factor 2 of its initial value down to r = 50");
        for (int k = 0; k < 5; k++) delete planes[k];
    }

    // (d) insensitivity to delta
    {
        RunResult A = run_schwarzschild(dist, 1e-3, 8.0, 20);
        RunResult B = run_schwarzschild(dist, 1e-5, 8.0, 20);
        map<pair<int,int>, double> zA;
        for (const CausticPoint& p : A.points) if (p.n == 1) zA[{p.ix, p.iy}] = p.X*n[0] + p.Y*n[1] + p.Z*n[2];
        double dmax = 0; int common = 0;
        for (const CausticPoint& p : B.points)
            if (p.n == 1 && zA.count({p.ix, p.iy}))
            {
                ++common;
                dmax = max(dmax, fabs(zA[{p.ix, p.iy}] - (p.X*n[0] + p.Y*n[1] + p.Z*n[2])));
            }
        cout << "  primary caustic position, delta = 1e-3 vs 1e-5: " << common << " rays compared, max |dz| = " << dmax << endl;
        check(common > 100 && dmax < 1e-3, "(d) primary caustic positions agree to < 1e-3 rg for delta = 1e-3 and 1e-5");
    }

    cout << endl << (n_fail == 0 ? "ALL CHECKS PASSED (0 failures)" : "CHECKS FAILED") << endl;
    return n_fail == 0 ? 0 : 1;
}
