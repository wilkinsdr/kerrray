//
// ray_transfer_kerr_vs_flat.cpp
//
// Isolates the effect of gravitational redshift on a disc-wind-style line spectrum, by tracing the same
// spherical wind + static corona twice: once through the full Kerr spacetime (RayTransfer -- by default
// with no disc, so the system stays spherically symmetric, the direct Kerr analogue of FlatRayTransfer's
// setup; optionally with the same DiscWithISCODestination ray_transfer_disc_wind.cpp uses, for an exact,
// pixel-for-pixel comparison against that application -- set r_out_disc > 0) and once through flat
// spacetime (FlatRayTransfer, straight-line rays, only the wind's own velocity field can shift the line --
// see docs/plan_ray_transfer.md for why this cannot instead be obtained as the spin -> 0 limit of the Kerr
// machinery: kerr.h fixes the central mass to 1 geometric unit everywhere, so "spin = 0" is Schwarzschild,
// not flat).  Both runs use identical wind (v_inf, beta, R0, R_out, n0) and corona/star (radius, intensity)
// parameters, so any systematic difference between the two spectra is gravitational redshift (plus, if
// spin != 0, frame dragging, and if the disc is enabled, disc blocking) rather than a difference in the
// wind physics itself. The flat side has no disc equivalent (FlatRayTransfer has no such concept), so
// enabling the disc only changes the Kerr side.
//
// Output: a single CSV (energy, kerr_residual, flat_residual) for direct overlay --
// python/ray_transfer_kerr_vs_flat_plot.py plots it.
//
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

using namespace std;

#include "ray_transfer.h"
#include "../raytracer/log_imageplane.h"
#include "../include/par_file.h"
#include "../include/par_args.h"

int main(int argc, char** argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile") ? par_args.get_string_parameter("--parfile")
                                                            : "../par/ray_transfer_kerr_vs_flat.par";
    ParameterFile par_file(par_filename.c_str());

    string out_filename = par_args.key_exists("--outfile") ? par_args.get_parameter<string>("--outfile")
                                                            : par_file.get_parameter<string>("outfile");

    // wind (shared by both runs)
    const double v_inf = par_file.get_parameter<double>("v_inf");
    const double v0 = par_file.get_parameter<double>("v0_vinf", 0.01) * v_inf;   // base velocity at R0
    const double beta_exp = par_file.get_parameter<double>("beta", 1.0);
    const double R0 = par_file.get_parameter<double>("R0");
    const double R_out = par_file.get_parameter<double>("R_out");
    const double n0 = par_file.get_parameter<double>("n0");

    // corona / star (shared by both runs)
    const double R_corona = par_file.get_parameter<double>("R_corona");
    const double I_corona = par_file.get_parameter<double>("I_corona", 1.0);

    // line (shared by both runs)
    const double line_energy = par_file.get_parameter<double>("line_energy", 1.0);
    const double doppler_width = par_file.get_parameter<double>("doppler_width");
    const double kappa0 = par_file.get_parameter<double>("kappa0");

    // energy grid (shared)
    const int n_energy = par_file.get_parameter<int>("n_energy", 90);
    const double energy_min = par_file.get_parameter<double>("energy_min");
    const double energy_max = par_file.get_parameter<double>("energy_max");

    // Kerr-side geometry.  spin = 0 (Schwarzschild) isolates gravitational redshift from frame dragging,
    // and then inclination/distance are irrelevant by spherical symmetry (no disc here) -- purely a
    // numerical choice of geometry.  spin != 0 (e.g. matched to ray_transfer_disc_wind.par for a direct,
    // pixel-for-pixel comparison between the two applications) brings frame dragging back in, so the
    // Kerr-vs-flat difference is then gravitational redshift *and* frame dragging together, not gravity
    // alone -- see docs/plan_ray_transfer.md Sec 5.3 vs Sec 5.7.
    const double spin = par_file.get_parameter<double>("spin", 0.0);
    const double dist = par_file.get_parameter<double>("dist", 1000.0);
    const double incl = par_file.get_parameter<double>("incl", 60.0);
    const double max_tstep = par_file.get_parameter<double>("max_tstep", 0.01);

    // grid_type: "linear" (default) is ImagePlane's original uniformly-spaced, square x0/xmax/Nx grid;
    // "log" is LogImagePlane's logarithmically-spaced grid (log_x_min/log_x_max/log_Nx/log_Nlinx and the
    // y-equivalents -- left at their x defaults, an un-overridden log grid stays square too, matching this
    // application's existing convention). See src/raytracer/log_imageplane.h and the pixel-grid query
    // methods in imageplane.h.
    const string grid_type = par_file.get_parameter<string>("grid_type", "linear");

    // Optional equatorial disc (ray_transfer_disc_wind.cpp's DiscWithISCODestination, inner edge at the
    // ISCO): r_out_disc <= 0 (default) means no disc, preserving the spin = 0 isolated-redshift
    // comparison (Sec 5.3); set it > 0 (matched to ray_transfer_disc_wind.par's value) for an exact,
    // pixel-for-pixel comparison against that application (Sec 5.7).
    const double r_out_disc = par_file.get_parameter<double>("r_out_disc", -1.0);
    const double r_isco = kerr_isco<double>(spin, +1);

    // Kerr-side wind line-emission source function -- see RayTransfer/WindSourceMode (ray_transfer.h) and
    // docs/plan_ray_transfer.md; matched to ray_transfer_disc_wind.par's convention for the same reason
    // everything else here is matched to it (Sec 5.7-5.9). The flat side has no equivalent switch: it
    // always uses the illumination-based I_star*dilution_factor(r,R_star) (FlatRayTransfer::trace_ray).
    const string source_mode_str = par_file.get_parameter<string>("source_mode", "illumination");
    const WindSourceMode source_mode = (source_mode_str == "density") ? WindSourceMode::Density
                                                                       : WindSourceMode::Illumination;
    const double density_scale = par_file.get_parameter<double>("density_scale", 1.0);

    // Flat-side geometry: impact parameter grid and line-of-sight integration range/step
    const double p_max = par_file.get_parameter<double>("p_max", 1.2 * R_out);
    const int n_p = par_file.get_parameter<int>("n_p", 300);
    const double z_max = par_file.get_parameter<double>("z_max", 1.5 * R_out);
    const double dz = par_file.get_parameter<double>("dz", 0.02 * R0);

    SphericalBetaWind<double> wind(v_inf, v0, beta_exp, R0, R_out, n0);
    LineTransition<double> line{line_energy, doppler_width, kappa0};
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(energy_min, energy_max, n_energy);

    // --- Kerr (Schwarzschild by default) run: sum over the image plane ---
    // Image-plane grid: see ray_transfer_disc_wind.cpp's identical block for the full rationale (in
    // particular why a single polymorphic unique_ptr<ImagePlane<double>> is safe now that ImagePlane has a
    // virtual destructor, and why RayTransfer's single constructor takes just this plane object).
    unique_ptr<ImagePlane<double>> plane;

    if (grid_type == "log")
    {
        const double log_x_min = par_file.get_parameter<double>("log_x_min");
        const double log_x_max = par_file.get_parameter<double>("log_x_max");
        const int log_Nx = par_file.get_parameter<int>("log_Nx");
        const int log_Nlinx = par_file.get_parameter<int>("log_Nlinx");
        const double log_y_min = par_file.get_parameter<double>("log_y_min", log_x_min);
        const double log_y_max = par_file.get_parameter<double>("log_y_max", log_x_max);
        const int log_Ny = par_file.get_parameter<int>("log_Ny", log_Nx);
        const int log_Nliny = par_file.get_parameter<int>("log_Nliny", log_Nlinx);

        plane = make_unique<LogImagePlane<double>>(dist, incl, log_x_min, log_x_max, log_Nx, log_Nlinx,
                                                     log_y_min, log_y_max, log_Ny, log_Nliny, spin, 0.0);
    }
    else
    {
        const double x0 = par_file.get_parameter<double>("x0", -1.2 * R_out);
        const double xmax = par_file.get_parameter<double>("xmax", 1.2 * R_out);
        const int Nx = par_file.get_parameter<int>("Nx", 60);
        const double dx = (xmax - x0) / Nx;

        plane = make_unique<ImagePlane<double>>(dist, incl, x0, xmax, dx, x0, xmax, dx, spin, 0.0);
    }

    SphericalContinuumSource<double> corona(R_corona, I_corona);
    DiscWithISCODestination<double> disc(r_isco, r_out_disc);
    const RayDestination<double>* disc_ptr = (r_out_disc > 0) ? &disc : nullptr;
    if (disc_ptr) cout << "Disc enabled: ISCO at " << r_isco << ", outer edge " << r_out_disc << endl;
    RayTransfer<double> rt(*plane, spin, wind, line, bins, disc_ptr, &corona, source_mode, density_scale);
    rt.set_max_tstep(max_tstep);   // far-field cap on the coordinate-time step (default MAXDT); symplectic
                                    // step/order left at their defaults (1/PRECISION, order 6)

    vector<double> kerr_line(n_energy, 0.0), kerr_total(n_energy, 0.0);
    double kerr_continuum_total = 0;
    long n_corona = 0;
    double kerr_tau_max = 0, kerr_tau_sum = 0;
    long kerr_n_thick = 0;

    const int Nx = rt.get_Nx(), Ny = rt.get_Ny();

    #pragma omp parallel for schedule(dynamic) reduction(+:kerr_continuum_total, n_corona) \
        reduction(max:kerr_tau_max) reduction(+:kerr_tau_sum, kerr_n_thick)
    for (int ix = 0; ix < Nx; ix++)
    {
        vector<double> line_emission(n_energy), absorption(n_energy);
        vector<double> row_line(n_energy, 0.0), row_total(n_energy, 0.0);
        const double x = plane->pixel_x(ix);
        for (int iy = 0; iy < Ny; iy++)
        {
            const double y = plane->pixel_y(iy);
            // Diagnostic-only accumulators (kerr_tau_max/kerr_tau_sum/kerr_n_thick below) are left as
            // plain per-pixel-count statistics regardless of grid_type; only the physically-integrated
            // flux/continuum sums are area-weighted -- see ImagePlane::pixel_weight()'s comment
            // (imageplane.h) for why grid_type=linear's weight is fixed at 1, not pixel_dx*pixel_dy.
            const double w = plane->pixel_weight(ix, iy);
            double continuum;
            if (rt.trace_pixel(x, y, line_emission, absorption, continuum)) ++n_corona;
            kerr_continuum_total += continuum * w;
            double tau_peak = 0;
            for (int j = 0; j < n_energy; j++)
            {
                const double flux = continuum * exp(-absorption[j]) + line_emission[j];
                row_line[j] += line_emission[j] * w;
                row_total[j] += flux * w;
                if (absorption[j] > tau_peak) tau_peak = absorption[j];
            }
            // peak optical depth over the energy grid for this sightline (absorption[j] is accumulated
            // over the whole backward-traced ray, ray_transfer.cpp:252-253, so this is the total wind
            // optical depth along the line of sight, not a per-step value) -- docs/plan_ray_transfer.md
            // Sec 5.15-5.16's thick-vs-thin diagnostic, exposed here too rather than only in a one-off script
            if (tau_peak > kerr_tau_max) kerr_tau_max = tau_peak;
            kerr_tau_sum += tau_peak;
            if (tau_peak > 1) ++kerr_n_thick;
        }
        #pragma omp critical
        for (int j = 0; j < n_energy; j++) { kerr_line[j] += row_line[j]; kerr_total[j] += row_total[j]; }
    }
    cout << "Kerr (spin=" << spin << "): " << n_corona << " / " << (Nx*Ny) << " pixels hit the corona" << endl;
    cout << "Kerr wind optical depth: peak tau = " << kerr_tau_max << ", mean tau = " << (kerr_tau_sum / (Nx*Ny))
         << ", " << kerr_n_thick << " / " << (Nx*Ny) << " sightlines optically thick (tau > 1)" << endl;

    // --- Flat run: sum over impact parameter (as ray_transfer_pcygni.cpp) ---
    FlatRayTransfer<double> flat_rt(wind, line, bins);
    vector<double> flat_total(n_energy, 0.0);
    double flat_continuum_total = 0;
    const double dp = p_max / n_p;
    double flat_tau_max = 0, flat_tau_sum = 0;
    long flat_n_thick = 0;

    // Self-normalise against the *same* discretised impact-parameter grid (rather than the exact analytic
    // pi*R_corona^2*I_corona), so the finite-dp quadrature error in resolving the corona/star's small
    // radius against a grid sized for the much larger wind cancels between numerator and denominator --
    // exactly how the Kerr side already self-normalises against its own discretised pixel sum.
    #pragma omp parallel for schedule(dynamic) reduction(+:flat_continuum_total) \
        reduction(max:flat_tau_max) reduction(+:flat_tau_sum, flat_n_thick)
    for (int i = 0; i < n_p; i++)
    {
        vector<double> em(n_energy), ab(n_energy);
        const double p = (i + 0.5) * dp;
        flat_rt.trace_ray(p, z_max, -z_max, dz, R_corona, I_corona, em, ab);
        const double weight = 2 * M_PI * p * dp;
        if (p < R_corona) flat_continuum_total += weight * I_corona;
        double tau_peak = 0;
        for (int j = 0; j < n_energy; j++)
            if (ab[j] > tau_peak) tau_peak = ab[j];
        if (tau_peak > flat_tau_max) flat_tau_max = tau_peak;
        flat_tau_sum += tau_peak;
        if (tau_peak > 1) ++flat_n_thick;
        #pragma omp critical
        for (int j = 0; j < n_energy; j++) flat_total[j] += weight * em[j];
    }
    cout << "Flat wind optical depth: peak tau = " << flat_tau_max << ", mean tau = " << (flat_tau_sum / n_p)
         << ", " << flat_n_thick << " / " << n_p << " sightlines optically thick (tau > 1)" << endl;

    ofstream csv(out_filename);
    csv << "energy,kerr_residual,flat_residual" << endl;
    for (int j = 0; j < n_energy; j++)
        csv << scientific << setprecision(8) << bins.energy[j] << ','
            << (kerr_continuum_total > 0 ? kerr_total[j] / kerr_continuum_total : 0) << ','
            << flat_total[j] / flat_continuum_total << endl;
    csv.close();
    cout << "Wrote " << out_filename << endl;
    return 0;
}
