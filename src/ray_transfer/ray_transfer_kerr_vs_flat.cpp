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

using namespace std;

#include "ray_transfer.h"
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
    const double x0 = par_file.get_parameter<double>("x0", -1.2 * R_out);
    const double xmax = par_file.get_parameter<double>("xmax", 1.2 * R_out);
    const int Nx = par_file.get_parameter<int>("Nx", 60);
    const double max_tstep = par_file.get_parameter<double>("max_tstep", 0.01);

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
    const double dx = (xmax - x0) / Nx;
    ImagePlane<double> plane(dist, incl, x0, xmax, dx, x0, xmax, dx, spin, 0.0);
    SphericalContinuumSource<double> corona(R_corona, I_corona);
    DiscWithISCODestination<double> disc(r_isco, r_out_disc);
    const RayDestination<double>* disc_ptr = (r_out_disc > 0) ? &disc : nullptr;
    if (disc_ptr) cout << "Disc enabled: ISCO at " << r_isco << ", outer edge " << r_out_disc << endl;
    RayTransfer<double> rt(plane, spin, wind, line, bins, Nx, Nx, x0, dx, x0, dx, disc_ptr, &corona,
                            source_mode, density_scale);
    rt.set_max_tstep(max_tstep);   // far-field cap on the coordinate-time step (default MAXDT); symplectic
                                    // step/order left at their defaults (1/PRECISION, order 6)

    vector<double> kerr_line(n_energy, 0.0), kerr_total(n_energy, 0.0);
    double kerr_continuum_total = 0;
    long n_corona = 0;
    double kerr_tau_max = 0, kerr_tau_sum = 0;
    long kerr_n_thick = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:kerr_continuum_total, n_corona) \
        reduction(max:kerr_tau_max) reduction(+:kerr_tau_sum, kerr_n_thick)
    for (int ix = 0; ix < Nx; ix++)
    {
        vector<double> line_emission(n_energy), absorption(n_energy);
        vector<double> row_line(n_energy, 0.0), row_total(n_energy, 0.0);
        const double x = x0 + (ix + 0.5) * dx;
        for (int iy = 0; iy < Nx; iy++)
        {
            const double y = x0 + (iy + 0.5) * dx;
            double continuum;
            if (rt.trace_pixel(x, y, line_emission, absorption, continuum)) ++n_corona;
            kerr_continuum_total += continuum;
            double tau_peak = 0;
            for (int j = 0; j < n_energy; j++)
            {
                const double flux = continuum * exp(-absorption[j]) + line_emission[j];
                row_line[j] += line_emission[j];
                row_total[j] += flux;
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
    cout << "Kerr (spin=" << spin << "): " << n_corona << " / " << (Nx*Nx) << " pixels hit the corona" << endl;
    cout << "Kerr wind optical depth: peak tau = " << kerr_tau_max << ", mean tau = " << (kerr_tau_sum / (Nx*Nx))
         << ", " << kerr_n_thick << " / " << (Nx*Nx) << " sightlines optically thick (tau > 1)" << endl;

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
