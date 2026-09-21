//
// ray_transfer_disc_wind.cpp
//
// X-ray line/continuum image of a spherical wind (inner/outer radius R0/R_out) around a Kerr black hole,
// illuminated by a static, spherical corona, with an equatorial accretion disc (ISCO to r_out_disc) that
// blocks rays -- built on RayTransfer<double> (src/ray_transfer/ray_transfer.h).  Per pixel: a scalar
// continuum (only for rays that terminate on the corona), an energy-dependent wind optical depth and line
// emission (accumulated along the whole ray regardless of its eventual fate).  Output: a FITS continuum
// map plus a (energy, x, y) flux data cube, flux = continuum*exp(-absorption) + line_emission; and a CSV
// spectrum (energy, line_flux, total_flux, residual) summed over the whole image plane -- the disc-wind
// analogue of the flat-space P-Cygni spectrum (ray_transfer_pcygni.cpp).
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
#include "../include/par_file.h"
#include "../include/par_args.h"
#include "../include/fits_output.h"
#include "../include/array.h"

int main(int argc, char** argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile") ? par_args.get_string_parameter("--parfile")
                                                            : "../par/ray_transfer_disc_wind.par";
    ParameterFile par_file(par_filename.c_str());

    string out_filename = par_args.key_exists("--outfile") ? par_args.get_parameter<string>("--outfile")
                                                            : par_file.get_parameter<string>("outfile");
    string spec_filename = par_args.key_exists("--specfile") ? par_args.get_parameter<string>("--specfile")
                          : par_file.get_parameter<string>("specfile", out_filename.substr(0, out_filename.rfind('.')) + "_spectrum.csv");

    // image plane, as in imageplane_disc_image.cpp
    const double dist = par_file.get_parameter<double>("dist");
    const double incl = (par_args.key_exists("--incl")) ? par_args.get_parameter<double>("--incl")
                                                         : par_file.get_parameter<double>("incl");
    const double spin = (par_args.key_exists("--spin")) ? par_args.get_parameter<double>("--spin")
                                                         : par_file.get_parameter<double>("spin");
    const double plane_phi0 = par_file.get_parameter<double>("plane_phi0", 0);
    const double x0 = par_file.get_parameter<double>("x0");
    const double xmax = par_file.get_parameter<double>("xmax");
    const int Nx = par_file.get_parameter<int>("Nx");
    const double y0 = par_file.get_parameter<double>("y0", x0);
    const double ymax = par_file.get_parameter<double>("ymax", xmax);
    const int Ny = par_file.get_parameter<int>("Ny", Nx);
    const double symp_step = par_file.get_parameter<double>("symp_step", -1);
    const int symp_order = par_file.get_parameter<int>("symp_order", 6);
    // Far-field step cap beyond r_cap = 2*horizon (mino_step_size, mino_stepper.h). The library-wide
    // default (MAXDT = 1) is tuned for cheaply tracing rays through the near-vacuum far field of a plain
    // disc image and is much too coarse to resolve a wind whose density varies steeply close to its own
    // launch radius R0 -- see docs/plan_ray_transfer.md Sec 6 (a coarse step there was found to alias into
    // a spurious, order-of-magnitude too large spike in the line spectrum; the default here is refined
    // enough that the same spike has converged to within ~10% of its finer-resolution value).
    const double max_tstep = par_file.get_parameter<double>("max_tstep", 0.01);

    // wind
    const double v_inf = par_file.get_parameter<double>("v_inf");
    // base velocity at R0: must be > 0 (not just numerically small) -- see SphericalBetaWind in
    // ray_transfer.h/docs/plan_ray_transfer.md for why a v = 0 launch point gives a divergent column
    // density that no amount of floor-tuning removes.
    const double v0 = par_file.get_parameter<double>("v0_vinf", 0.01) * v_inf;
    const double beta_exp = par_file.get_parameter<double>("beta", 1.0);
    const double R0 = par_file.get_parameter<double>("R0");
    const double R_out = par_file.get_parameter<double>("R_out");
    const double n0 = par_file.get_parameter<double>("n0");
    // wind latitude range: "spherical" (default) fills the whole sphere (SphericalBetaWind); "disc" and
    // "polar" restrict it to a range of polar angle (ConicalBetaWind, ray_transfer.h) -- "disc" hugs the
    // disc surface (hollow near both poles), "polar" is collimated along the polar axis (hollow near the
    // equator); theta_lim (degrees, measured from the pole) is required for either. See
    // docs/plan_ray_transfer.md Sec 5.22-5.23.
    const string wind_geometry = par_file.get_parameter<string>("wind_geometry", "spherical");

    // corona
    const double R_corona = par_file.get_parameter<double>("R_corona");
    const double I_corona = par_file.get_parameter<double>("I_corona", 1.0);

    // disc (inner edge: ISCO)
    const double r_out_disc = par_file.get_parameter<double>("r_out_disc");

    // line
    const double line_energy = par_file.get_parameter<double>("line_energy", 1.0);
    const double doppler_width = par_file.get_parameter<double>("doppler_width");
    const double kappa0 = par_file.get_parameter<double>("kappa0");

    // energy grid
    const int n_energy = par_file.get_parameter<int>("n_energy", 50);
    const double energy_min = par_file.get_parameter<double>("energy_min");
    const double energy_max = par_file.get_parameter<double>("energy_max");

    const double r_isco = kerr_isco<double>(spin, +1);
    cout << "ISCO at " << r_isco << ", corona radius " << R_corona << endl;

    // Wind line-emission source function: "illumination" (default) ties it to the corona's actual
    // brightness at each point (ContinuumSource::illumination, continuum_source.h); "density" is a simple,
    // quick placeholder (density_scale * density) with no such tie -- see docs/plan_ray_transfer.md.
    const string source_mode_str = par_file.get_parameter<string>("source_mode", "illumination");
    const WindSourceMode source_mode = (source_mode_str == "density") ? WindSourceMode::Density
                                                                       : WindSourceMode::Illumination;
    const double density_scale = par_file.get_parameter<double>("density_scale", 1.0);
    cout << "Wind source function: " << source_mode_str
         << (source_mode == WindSourceMode::Density ? (" (density_scale = " + to_string(density_scale) + ")") : "")
         << endl;

    const double dx = (xmax - x0) / Nx;
    const double dy = (ymax - y0) / Ny;

    ImagePlane<double> plane(dist, incl, x0, xmax, dx, y0, ymax, dy, spin, plane_phi0);

    unique_ptr<RTField<double>> wind_ptr;
    if (wind_geometry == "disc" || wind_geometry == "polar")
    {
        const double theta_lim = par_file.get_parameter<double>("theta_lim") * M_PI / 180.0;
        const WindLatitudeMode mode = (wind_geometry == "disc") ? WindLatitudeMode::DiscLaunched
                                                                 : WindLatitudeMode::PolarCollimated;
        wind_ptr = make_unique<ConicalBetaWind<double>>(v_inf, v0, beta_exp, R0, R_out, n0, theta_lim, mode);
        cout << "Wind geometry: " << wind_geometry << ", theta_lim = " << par_file.get_parameter<double>("theta_lim")
             << " deg" << endl;
    }
    else
    {
        wind_ptr = make_unique<SphericalBetaWind<double>>(v_inf, v0, beta_exp, R0, R_out, n0);
        cout << "Wind geometry: spherical" << endl;
    }
    RTField<double>& wind = *wind_ptr;
    SphericalContinuumSource<double> corona(R_corona, I_corona);
    DiscWithISCODestination<double> disc(r_isco, r_out_disc);
    LineTransition<double> line{line_energy, doppler_width, kappa0};
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(energy_min, energy_max, n_energy);

    RayTransfer<double> rt(plane, spin, wind, line, bins, Nx, Ny, x0, dx, y0, dy, &disc, &corona,
                            source_mode, density_scale);
    rt.set_symplectic_step(symp_step);
    rt.set_symplectic_order(symp_order);
    rt.set_max_tstep(max_tstep);   // far-field cap on the coordinate-time step (default MAXDT)

    // Traces every pixel and fills rt.continuum_map/tau_map/flux_cube/spec_line/spec_total/
    // continuum_total/n_corona/tau_corona_max in place (RayTransfer::run_raytrace, ray_transfer.h/.cpp) --
    // tau_map is the standard "how thick is this sightline" diagnostic: since absorption[j] is
    // accumulated over the *whole* backward-traced ray (near side, periapsis, far side alike,
    // ray_transfer.cpp) and never reset, tau_map[ix][iy] is genuinely the total wind optical depth along
    // that line of sight, not just a per-step or per-leg value (docs/plan_ray_transfer.md Sec 5.15-5.16).
    rt.run_raytrace();
    cout << rt.n_corona << " / " << (Nx * Ny) << " pixels hit the corona" << endl;

    // Optically-thick-vs-thin summary: tau_map's peak, mean, and the fraction of sightlines with
    // tau_peak > 1 (the standard optically-thick threshold) -- a quick read on the regime for this
    // parameter choice, without needing a separate diagnostic run.
    {
        double tau_max = 0, tau_sum = 0;
        long n_thick = 0;
        for (int ix = 0; ix < Nx; ix++)
            for (int iy = 0; iy < Ny; iy++)
            {
                const double t = (*rt.tau_map)[ix][iy];
                tau_sum += t;
                if (t > tau_max) tau_max = t;
                if (t > 1) ++n_thick;
            }
        const long n_pix = (long)Nx * Ny;
        cout << "Wind optical depth: peak tau = " << tau_max << ", mean tau = " << (tau_sum / n_pix)
             << ", " << n_thick << " / " << n_pix << " sightlines optically thick (tau > 1)" << endl;
        cout << "Peak optical depth to the corona (absorption trough depth against the continuum): "
             << rt.tau_corona_max << endl;
    }

    // Image-plane-integrated spectrum, normalised to the (energy-independent) unabsorbed continuum level
    // summed over the same pixels -- the disc-wind analogue of ray_transfer_pcygni's residual flux.
    ofstream spec_csv(spec_filename);
    spec_csv << "energy,line_flux,total_flux,residual" << endl;
    for (int j = 0; j < n_energy; j++)
        spec_csv << scientific << setprecision(8) << bins.energy[j] << ',' << rt.spec_line[j] << ','
                 << rt.spec_total[j] << ',' << (rt.continuum_total > 0 ? rt.spec_total[j] / rt.continuum_total : 0) << endl;
    spec_csv.close();
    cout << "Wrote " << spec_filename << endl;

    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("RayTransfer disc-wind continuum and line/continuum flux cube");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "ray_transfer_disc_wind");
    fits.write_keyword("DIST", "Distance to image plane", dist);
    fits.write_keyword("INCL", "Inclination of line of sight", incl);
    fits.write_keyword("SPIN", "Black hole spin", spin);
    fits.write_keyword("ISCO", "Innermost stable circular orbit", r_isco);
    fits.write_keyword("RDISCOUT", "Outer radius of disc", r_out_disc);
    fits.write_keyword("RCORONA", "Corona radius", R_corona);
    fits.write_keyword("R0", "Wind inner radius", R0);
    fits.write_keyword("ROUT", "Wind outer radius", R_out);
    fits.write_keyword("VINF", "Wind terminal velocity (c)", v_inf);
    fits.write_keyword("LINEEN", "Line rest energy", line_energy);
    fits.write_keyword("NCORONA", "Pixels that hit the corona", rt.n_corona);

    fits.write_image(*rt.continuum_map, Nx, Ny, false);
    fits.set_ext_name("CONTINUUM");
    fits.write_comment("Unabsorbed corona continuum (0 where the ray did not reach the corona)");

    fits.write_image(*rt.tau_map, Nx, Ny, false);
    fits.set_ext_name("TAU");
    fits.write_comment("Peak wind optical depth over the energy grid, per line of sight (max_j absorption[j])");

    fits.write_me_data_cube(rt.flux_cube->ptr, n_energy, Nx, Ny);
    fits.set_ext_name("FLUX");
    fits.write_comment("continuum*exp(-absorption) + line_emission per energy bin");
    fits.write_keyword("EMIN", "Minimum energy", energy_min);
    fits.write_keyword("EMAX", "Maximum energy", energy_max);
    fits.write_keyword("NE", "Number of energy bins", n_energy);

    fits.close();
    cout << "Wrote " << out_filename << endl;
    return 0;
}
