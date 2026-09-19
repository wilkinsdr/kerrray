//
// caustic_bundle_test.cpp
//
// Consistency checks of CausticBundle (caustic_bundle.h) against the raytracer library it is built on:
//   (a) ImagePlane::init_ray reproduces init_image_plane exactly (grid pixels and off-grid positions)
//   (b) CausticBundle::emit() reproduces Raytracer::redshift_start() exactly on the grid
//   (c) a bundle built at a grid pixel and one built at the same (x, y) off the grid trace identically, and
//       tracing is deterministic (two traces of the same bundle agree bitwise)
//
// History: when the class replaced the free-function tracer (2026-09-18) this test also compared every bundle
// against the legacy implementation (kept as caustic_bundle_legacy.h at the time): bitwise-identical status,
// counts, caustic lists and disc crossings for 2 x 1881 bundles; see docs/plan_caustic_bundle_class.md.
//
// Build (from the project root, no cfitsio needed):
//   g++ -O2 -std=c++17 -fopenmp -I src src/tests/caustic_bundle_test.cpp src/raytracer/raytracer.cpp \
//       src/raytracer/imageplane.cpp -o bin/caustic_bundle_test
//
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

#include "raytracer/imageplane.h"
#include "caustic/caustic_bundle.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& msg)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << msg << endl;
    if (!ok) ++failures;
}
static bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

static bool same_results(const CausticBundle<double>& a, const CausticBundle<double>& b)
{
    const PixelResult& ra = a.result(); const PixelResult& rb = b.result();
    if (ra.status != rb.status || ra.ncaust != rb.ncaust || ra.eqcross != rb.eqcross || !same(ra.tau_end, rb.tau_end)) return false;
    if (a.caustics().size() != b.caustics().size() || a.disc_crossings().size() != b.disc_crossings().size()) return false;
    for (size_t i = 0; i < a.caustics().size(); i++)
    {
        const CausticPoint& p = a.caustics()[i]; const CausticPoint& q = b.caustics()[i];
        if (p.n != q.n || !same(p.tau, q.tau) || !same(p.r, q.r) || !same(p.theta, q.theta) || !same(p.phi, q.phi)
            || p.rflips != q.rflips || p.eqcross != q.eqcross || p.dj_sign != q.dj_sign || !same(p.logscale, q.logscale)) return false;
    }
    for (size_t i = 0; i < a.disc_crossings().size(); i++)
    {
        const DiscCrossing& p = a.disc_crossings()[i]; const DiscCrossing& q = b.disc_crossings()[i];
        if (p.n != q.n || !same(p.J, q.J) || !same(p.r, q.r) || !same(p.phi, q.phi) || !same(p.t, q.t) || p.ncaust != q.ncaust) return false;
    }
    return true;
}

int main()
{
    const double spin = 0.998, incl = 60, dist = 1000, delta = 1e-4;
    const double x0 = -8, xmax = 8; const int Nx = 40;
    const double dx = (xmax - x0) / Nx, dx_ip = dx * (1 - 1e-9);

    ImagePlane<double> plane(dist, incl, x0, xmax, dx_ip, x0, xmax, dx_ip, spin, 0, PRECISION);
    plane.redshift_start();
    const int nPix = plane.get_count();

    cout << "\n(a) init_ray vs init_image_plane" << endl;
    {
        double mx = 0; bool signs_ok = true;
        mt19937 rng(7); uniform_real_distribution<double> U(x0, xmax);
        for (int k = 0; k < nPix + 200; k++)
        {
            double x, y;
            if (k < nPix) { x = plane.rays[k].alpha; y = plane.rays[k].beta; } else { x = U(rng); y = U(rng); }
            ImagePlane<double> single(dist, incl, x, x, 1, y, y, 1, spin, 0, PRECISION);
            Ray<double> r2; plane.init_ray(r2, x, y);
            const Ray<double>& r1 = (k < nPix) ? plane.rays[k] : single.rays[0];
            const double d[] = { fabs(r1.r - r2.r), fabs(r1.theta - r2.theta), fabs(r1.phi - r2.phi), fabs(r1.k - r2.k),
                                 fabs(r1.h - r2.h), fabs(r1.Q - r2.Q), fabs(r1.pt - r2.pt), fabs(r1.pr - r2.pr), fabs(r1.ptheta - r2.ptheta),
                                 fabs(r1.pphi - r2.pphi), fabs(r1.alpha - r2.alpha), fabs(r1.beta - r2.beta) };
            for (double v : d) if (!(v <= mx)) mx = std::isnan(v) ? (std::isnan(r1.h) && std::isnan(r2.h) ? mx : 1e99) : v;
            if (r1.rdot_sign != r2.rdot_sign || r1.thetadot_sign != r2.thetadot_sign) signs_ok = false;
        }
        cout << "  " << nPix << " grid pixels + 200 random positions: max |diff| = " << scientific << mx << fixed << endl;
        check(mx == 0 && signs_ok, "(a) init_ray reproduces init_image_plane exactly");
    }

    cout << "\n(b) emit() vs redshift_start()" << endl;
    {
        CausticBundle<double>::Params P = CausticBundle<double>::Params::from_plane(plane, spin, delta, -1, PRECISION);
        double mx = 0;
        for (int pix = 0; pix < nPix; pix++)
        {
            CausticBundle<double> b(plane, P, plane.rays[pix].alpha, plane.rays[pix].beta);
            if (!b.valid()) continue;
            mx = max(mx, fabs(b.emit() - plane.rays[pix].emit));
        }
        cout << "  max |diff| = " << scientific << mx << fixed << endl;
        check(mx == 0, "(b) emit() reproduces redshift_start() exactly");
    }

    cout << "\n(c) determinism and grid / off-grid construction" << endl;
    {
        CausticBundle<double>::Params P = CausticBundle<double>::Params::from_plane(plane, spin, delta, -1, PRECISION);
        P.max_eqcross = 3;
        int traced = 0, mismatch = 0; long ncaust = 0, ndisc = 0;
        #pragma omp parallel for schedule(dynamic) reduction(+:traced,mismatch,ncaust,ndisc)
        for (int pix = 0; pix < nPix; pix++)
        {
            CausticBundle<double> b1(plane, P, plane.rays[pix].alpha, plane.rays[pix].beta, plane.get_x_index(pix), plane.get_y_index(pix));
            CausticBundle<double> b2(plane, P, plane.rays[pix].alpha, plane.rays[pix].beta);   // no labels: same rays
            if (!b1.valid()) continue;
            b1.trace(); b2.trace();
            CausticBundle<double> b3 = b1; b3.trace();                                        // re-trace a traced copy
            ++traced;
            if (!same_results(b1, b2) || !same_results(b1, b3)) ++mismatch;
            ncaust += b1.caustics().size(); ndisc += b1.disc_crossings().size();
        }
        cout << "  " << traced << " bundles, " << ncaust << " caustic points, " << ndisc << " disc crossings, " << mismatch << " mismatches" << endl;
        check(traced > 1000 && mismatch == 0 && ncaust > 1000 && ndisc > 500, "(c) identical results for identical initial data; re-tracing a bundle is deterministic");
    }

    cout << endl;
    if (failures == 0) cout << "ALL CHECKS PASSED (0 failures)" << endl;
    else cout << failures << " CHECK(S) FAILED" << endl;
    return failures == 0 ? 0 : 1;
}
