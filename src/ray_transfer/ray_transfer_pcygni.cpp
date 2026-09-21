//
// ray_transfer_pcygni.cpp
//
// P-Cygni line profile from a spherical wind with no black hole (flat spacetime), using
// FlatRayTransfer.  Sums the per-impact-parameter emission/absorption spectra (FlatRayTransfer::trace_ray)
// weighted by projected annulus area into a continuum-normalised residual flux spectrum.
//
#include <iostream>
#include <iomanip>
#include <fstream>
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
                                                            : "../par/ray_transfer_pcygni.par";
    ParameterFile par_file(par_filename.c_str());

    string out_filename = par_args.key_exists("--outfile") ? par_args.get_parameter<string>("--outfile")
                                                            : par_file.get_parameter<string>("outfile");

    const double v_inf = par_file.get_parameter<double>("v_inf");
    const double v0 = par_file.get_parameter<double>("v0_vinf", 0.01) * v_inf;   // base velocity at R0
    const double beta_exp = par_file.get_parameter<double>("beta", 1.0);
    const double R0 = par_file.get_parameter<double>("R0", 1.0);
    const double R_out = par_file.get_parameter<double>("R_out_R0", 1e6) * R0;   // effectively unbounded by default
    const double n0 = par_file.get_parameter<double>("n0", 1.0);
    const double kappa0 = par_file.get_parameter<double>("kappa0");
    const double line_energy = par_file.get_parameter<double>("line_energy", 1.0);
    const double doppler_width = par_file.get_parameter<double>("doppler_width");
    const double I_star = par_file.get_parameter<double>("I_star", 1.0);

    const double p_max = par_file.get_parameter<double>("p_max_R0", 5.0) * R0;
    const int n_p = par_file.get_parameter<int>("n_p", 200);
    const double z_max = par_file.get_parameter<double>("z_max_R0", 20.0) * R0;
    const double dz = par_file.get_parameter<double>("dz_R0", 0.01) * R0;

    const int n_energy = par_file.get_parameter<int>("n_energy", 200);
    const double energy_min = par_file.get_parameter<double>("energy_min");
    const double energy_max = par_file.get_parameter<double>("energy_max");

    LineTransition<double> line{line_energy, doppler_width, kappa0};
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(energy_min, energy_max, n_energy);
    SphericalBetaWind<double> wind(v_inf, v0, beta_exp, R0, R_out, n0);
    FlatRayTransfer<double> rt(wind, line, bins);

    vector<double> flux(n_energy, 0.0), em(n_energy), ab(n_energy);
    const double dp = p_max / n_p;

    cout << "Tracing " << n_p << " impact parameters..." << endl;
    for (int i = 0; i < n_p; i++)
    {
        const double p = (i + 0.5) * dp;
        rt.trace_ray(p, z_max, -z_max, dz, R0, I_star, em, ab);
        const double weight = 2 * M_PI * p * dp;   // annulus area element
        for (int j = 0; j < n_energy; j++)
            flux[j] += weight * em[j];
    }

    // flux that would be observed with no wind at all (uniform stellar disc of intensity I_star)
    const double norm = M_PI * R0 * R0 * I_star;

    ofstream csv(out_filename);
    csv << "energy,flux,residual" << endl;
    for (int j = 0; j < n_energy; j++)
        csv << scientific << setprecision(8) << bins.energy[j] << ',' << flux[j] << ',' << flux[j] / norm << endl;
    csv.close();

    cout << "Wrote " << n_energy << " energy bins to " << out_filename << endl;
    return 0;
}
