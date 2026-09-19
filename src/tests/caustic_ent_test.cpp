//
// caustic_ent_test.cpp
//
// Self-checking tests for the pieces of caustic_ent that do not need cfitsio:
//   (a) the disc-crossing record of the bundle tracer agrees with the raytracer's own landing on
//       theta = pi/2 (r, phi, t) and with ImagePlane::redshift for the same pixel (Kerr, a = 0.998, i = 60 deg)
//   (b) face-on Schwarzschild: the disc redshift of the direct image equals sqrt(1 - 3/r)
//   (c) the equatorial-crossing Jacobian is continuous with the 3D Jacobian record (a bundle crossing
//       the plane twice has J of the same sign at the crossing as just before/after it when no caustic
//       is crossed in between)
//   (d) zero_contours recovers a circle (radius to O(dx^2)), a straight line, and leaves a NaN hole open
//
// Build (from the project root, no cfitsio needed):
//   g++ -O2 -std=c++17 -fopenmp -I src src/tests/caustic_ent_test.cpp src/raytracer/raytracer.cpp \
//       src/raytracer/imageplane.cpp -o bin/caustic_ent_test
//
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <limits>

#include "raytracer/imageplane.h"
#include "caustic/caustic_bundle.h"
#include "caustic/disc_caustic_contour.h"
#include "caustic/adaptive_plane.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& msg)
{
    cout << (ok ? "  [PASS] " : "  [FAIL] ") << msg << endl;
    if (!ok) ++failures;
}

struct PlaneSet
{
    ImagePlane<double>* plane;
    CausticBundle<double>::Params P;
    int nPix, img_Nx, img_Ny;
    double x0, y0, dx, dy;
    // a traced bundle for pixel pix (nullptr-safe: invalid bundles are returned untraced)
    CausticBundle<double> bundle(int pix) const
    {
        CausticBundle<double> b(*plane, P, plane->rays[pix].alpha, plane->rays[pix].beta,
                                plane->get_x_index(pix), plane->get_y_index(pix));
        if (b.valid()) b.trace();
        return b;
    }
};

static PlaneSet make_planes(double dist, double incl, double spin, double x0, double xmax, int Nx, double delta,
                            int max_eqcross)
{
    PlaneSet S;
    const double dx = (xmax - x0) / Nx, dx_ip = dx * (1 - 1e-9);
    S.plane = new ImagePlane<double>(dist, incl, x0, xmax, dx_ip, x0, xmax, dx_ip, spin, 0, PRECISION);
    S.plane->redshift_start();
    S.img_Nx = S.img_Ny = Nx + 1;
    S.nPix = S.img_Nx * S.img_Ny;
    S.x0 = S.y0 = x0; S.dx = S.dy = dx;
    S.P = CausticBundle<double>::Params::from_plane(*S.plane, spin, delta, -1, PRECISION);
    S.P.max_eqcross = max_eqcross;
    return S;
}

int main()
{
    cout << fixed << setprecision(6);

    // ------------------------------------------------------------------------------------------------
    // (a) Kerr consistency with the raytracer's own disc landing
    // ------------------------------------------------------------------------------------------------
    cout << "\n(a) disc crossing vs Raytracer landing and ImagePlane::redshift (a = 0.998, i = 60)" << endl;
    {
        const double spin = 0.998, incl = 60, dist = 1000;
        PlaneSet S = make_planes(dist, incl, spin, -8, 8, 16, 1e-4, 1);

        // independent landing with the raytracer's own symplectic propagator (stops on theta = pi/2)
        ImagePlane<double> ref(dist, incl, -8, 8, 1.0 * (1 - 1e-9), -8, 8, 1.0 * (1 - 1e-9), spin, 0, PRECISION);
        ref.redshift_start();
        ref.run_raytrace(Integrator::Symplectic, M_PI_2, 1.1 * dist, 0);
        ref.redshift(-1.0, true);

        double max_dr = 0, max_dphi = 0, max_dt = 0, max_dg = 0;
        int compared = 0;
        for (int pix = 0; pix < S.nPix; pix++)
        {
            CausticBundle<double> b = S.bundle(pix);
            if (!b.valid()) continue;
            const vector<DiscCrossing>& dc = b.disc_crossings();
            const Ray<double>& R = ref.rays[pix];
            if (dc.empty() || !(R.status & RAY_STATUS_DEST) || fabs(R.theta - M_PI_2) > 1e-6) continue;
            if (dc[0].r < 1.05 * kerr_horizon<double>(spin)) continue;
            const double g = b.disc_redshift(dc[0]);
            max_dr   = max(max_dr,   fabs(dc[0].r - R.r) / R.r);
            max_dphi = max(max_dphi, fabs(atan2(sin(dc[0].phi - R.phi), cos(dc[0].phi - R.phi))));
            max_dt   = max(max_dt,   fabs(dc[0].t - R.t));
            max_dg   = max(max_dg,   fabs(g - R.redshift) / R.redshift);
            ++compared;
        }
        cout << "  " << compared << " pixels compared: max |dr|/r = " << scientific << max_dr
             << ", max |dphi| = " << max_dphi << ", max |dt| = " << max_dt << ", max |dg|/g = " << max_dg << fixed << endl;
        check(compared > 100, "enough rays land on the disc");
        check(max_dr < 1e-6 && max_dphi < 1e-6 && max_dt < 1e-4, "(a1) landing r, phi, t agree with the raytracer");
        check(max_dg < 1e-8, "(a2) disc redshift agrees with ImagePlane::redshift");
        delete S.plane;
    }

    // ------------------------------------------------------------------------------------------------
    // (b) face-on Schwarzschild: g = sqrt(1 - 3/r)
    // ------------------------------------------------------------------------------------------------
    cout << "\n(b) face-on Schwarzschild disc redshift vs sqrt(1 - 3/r)" << endl;
    {
        const double spin = 0, incl = 1e-3, dist = 1000;
        PlaneSet S = make_planes(dist, incl, spin, -12, 12, 24, 1e-4, 1);
        double max_err = 0; int compared = 0;
        for (int pix = 0; pix < S.nPix; pix++)
        {
            CausticBundle<double> b = S.bundle(pix);
            if (!b.valid()) continue;
            const vector<DiscCrossing>& dc = b.disc_crossings();
            if (dc.empty() || dc[0].r < 6) continue;
            const double g = b.disc_redshift(dc[0]);
            // E_disc / E_obs = u^t sqrt(1 - 2/dist): the image plane sits at finite distance, where the
            // static observer's energy is E / sqrt(1 - 2/dist)
            const double g_exact = sqrt(1 - 2 / dist) / sqrt(1 - 3 / dc[0].r);
            max_err = max(max_err, fabs(g - g_exact) / g_exact);
            ++compared;
        }
        cout << "  " << compared << " pixels with r >= 6: max relative error " << scientific << max_err << fixed << endl;
        check(compared > 50 && max_err < 1e-4, "(b) E_disc/E_obs = sqrt(1 - 2/dist) / sqrt(1 - 3/r) for the face-on direct image");
        delete S.plane;
    }

    // ------------------------------------------------------------------------------------------------
    // (c) J at the crossing is consistent with the crossing count of caustics along the ray: sign(J_disc)
    //     equals (-1)^ncaust times sign(J at the image plane), i.e. J flips sign exactly at the recorded caustics
    // ------------------------------------------------------------------------------------------------
    cout << "\n(c) sign of J at the disc crossing vs caustic count along the ray (a = 0.998, i = 60)" << endl;
    {
        const double spin = 0.998, incl = 60, dist = 1000;
        PlaneSet S = make_planes(dist, incl, spin, -8, 8, 32, 1e-4, 3);
        int consistent = 0, total = 0;
        for (int pix = 0; pix < S.nPix; pix++)
        {
            CausticBundle<double> b(*S.plane, S.P, S.plane->rays[pix].alpha, S.plane->rays[pix].beta);
            if (!b.valid()) continue;
            const double J0 = b.jacobian();
            b.trace();
            for (auto& d : b.disc_crossings())
            {
                if (!isfinite(d.J) || d.J == 0) continue;
                const int expect = ((d.ncaust % 2) == 0) ? 1 : -1;
                const int got = (d.J * J0 > 0) ? 1 : -1;
                ++total;
                if (got == expect) ++consistent;
            }
        }
        cout << "  " << consistent << " / " << total << " crossings have sign(J) = (-1)^ncaust sign(J_plane)" << endl;
        check(total > 500 && consistent == total, "(c) J at the disc crossings flips sign exactly at the recorded caustics");
        delete S.plane;
    }

    // ------------------------------------------------------------------------------------------------
    // (d) zero_contours on synthetic fields
    // ------------------------------------------------------------------------------------------------
    cout << "\n(d) zero contours of synthetic fields" << endl;
    {
        const int nx = 101, ny = 101;
        const double x0 = -5, y0 = -5, dx = 0.1, dy = 0.1;
        // circle of radius 3 with a NaN hole on the left, and a line x = 1.234 elsewhere ... use two fields
        vector<double> circ(nx * ny), line(nx * ny), hole(nx * ny);
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++)
            {
                const double x = x0 + ix * dx, y = y0 + iy * dy;
                circ[iy*nx + ix] = x*x + y*y - 9;
                line[iy*nx + ix] = x - 1.234;
                hole[iy*nx + ix] = (x < -2.5) ? numeric_limits<double>::quiet_NaN() : x*x + y*y - 9;
            }
        auto C = zero_contours(circ, nx, ny, x0, y0, dx, dy);
        double max_dr = 0; size_t nv = 0;
        for (auto& c : C) for (auto& v : c.vertices) { max_dr = max(max_dr, fabs(sqrt(v.x*v.x + v.y*v.y) - 3)); ++nv; }
        cout << "  circle: " << C.size() << " curve(s), " << nv << " vertices, max |r - 3| = " << scientific << max_dr << fixed
             << (C.size() == 1 && C[0].closed ? " (closed)" : " (open)") << endl;
        check(C.size() == 1 && C[0].closed && max_dr < 2e-3, "(d1) circle: one closed curve, radius to O(dx^2)");

        auto L = zero_contours(line, nx, ny, x0, y0, dx, dy);
        double max_dx = 0; nv = 0;
        for (auto& c : L) for (auto& v : c.vertices) { max_dx = max(max_dx, fabs(v.x - 1.234)); ++nv; }
        cout << "  line: " << L.size() << " curve(s), " << nv << " vertices, max |x - 1.234| = " << scientific << max_dx << fixed << endl;
        check(L.size() == 1 && !L[0].closed && nv == static_cast<size_t>(ny) && max_dx < 1e-12, "(d2) line: one open curve through every row, exact position");

        auto H = zero_contours(hole, nx, ny, x0, y0, dx, dy);
        size_t nh = 0; bool any_closed = false; double min_x = 1e9;
        for (auto& c : H) { nh += c.vertices.size(); any_closed |= c.closed; for (auto& v : c.vertices) min_x = min(min_x, v.x); }
        cout << "  hole: " << H.size() << " curve(s), " << nh << " vertices, min x = " << min_x << endl;
        check(H.size() == 1 && !any_closed && min_x > -2.6 && nh < nv * 2, "(d3) NaN region: one open arc, no vertices inside the hole");
    }

    // ------------------------------------------------------------------------------------------------
    // (e) adaptive refinement: circle contour on a coarse grid refined 3 levels around the contour
    // ------------------------------------------------------------------------------------------------
    cout << "\n(e) adaptive refinement of the plane around a circle contour" << endl;
    {
        const int Nx = 20, Ny = 20, levels = 3;
        const double x0 = -5, y0 = -5, dx = 0.5, dy = 0.5;
        AdaptivePlane ap(x0, y0, dx, dy, Nx, Ny, levels);
        auto fval = [](double x, double y) { return x*x + y*y - 9; };
        vector<double> vals;
        auto eval_new = [&](const vector<int>& idx) { for (int k : idx) { vals.resize(ap.npoints()); vals[k] = fval(ap.point(k).x, ap.point(k).y); } };
        vector<int> base(ap.npoints()); for (int k = 0; k < ap.npoints(); k++) base[k] = k;
        eval_new(base);
        double err_prev = 1e9; bool improving = true; size_t npts_prev = 0;
        for (int l = 0; l < levels; l++)
        {
            vector<int> created = ap.refine(l, [&](const PlaneCell& c)
            {
                for (int e = 0; e < 4; e++)
                {
                    const int p = c.corner[e], q = c.corner[(e + 1) % 4];
                    if ((vals[p] >= 0) != (vals[q] >= 0)) return true;
                }
                return false;
            });
            eval_new(created);
            auto C = zero_contours_cells(ap.leaf_cells(), vals, ap.points());
            double max_dr = 0; size_t nv = 0;
            for (auto& c : C) for (auto& v : c.vertices) { max_dr = max(max_dr, fabs(hypot(v.x, v.y) - 3)); ++nv; }
            cout << "  level " << l + 1 << ": " << ap.npoints() << " points (" << created.size() << " new), "
                 << C.size() << " curve(s), " << nv << " vertices, max |r - 3| = " << scientific << max_dr << fixed
                 << ((C.size() == 1 && C[0].closed) ? " (closed)" : " (open)") << endl;
            if (!(C.size() == 1 && C[0].closed)) improving = false;
            if (max_dr > 0.35 * err_prev) improving = false;   // O(h^2): a factor 4 per level, allow 0.35
            err_prev = max_dr; npts_prev = ap.npoints();
        }
        const int full = (Nx * (1 << levels) + 1) * (Ny * (1 << levels) + 1);
        cout << "  " << npts_prev << " points vs " << full << " for a uniform grid at the finest level" << endl;
        check(improving, "(e1) one closed curve at every level, error falling by > 1/0.35 per level");
        check(npts_prev < full / 4, "(e2) refined plane uses < 1/4 of the points of the uniform fine grid");
    }

    cout << endl;
    if (failures == 0) cout << "ALL CHECKS PASSED (0 failures)" << endl;
    else cout << failures << " CHECK(S) FAILED" << endl;
    return failures == 0 ? 0 : 1;
}
