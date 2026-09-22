//
// ray_transfer_disc_surface.cpp
//
// X-ray line/continuum image of a Keplerian, co-rotating cylindrical gas layer sitting above the surface
// of a Kerr accretion disc (DiscSurface, rtfield.h: cylindrical radius R_in..R_out, height z_min..z_max
// above the equatorial plane, purely azimuthal Keplerian velocity), illuminated by an emitting annulus of
// the disc surface itself (DiscContinuumSource, continuum_source.h: R_in_corona..R_out_corona), with the
// rest of the disc (ISCO to r_out_disc) opaque but non-emitting -- built on RayTransfer<double>
// (src/ray_transfer/ray_transfer.h), the disc-surface analogue of ray_transfer_disc_wind.cpp (spherical
// wind + compact corona). Per pixel: a scalar continuum (only for rays that terminate on the disc's
// emitting annulus), an energy-dependent optical depth and line emission through the gas layer
// (accumulated along the whole ray regardless of its eventual fate). Output: a FITS continuum map plus a
// (energy, x, y) flux data cube, flux = continuum*exp(-absorption) + line_emission; and a CSV spectrum
// (energy, line_flux, total_flux, residual) summed over the whole image plane.
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
#include "../include/fits_output.h"
#include "../include/array.h"

int main(int argc, char** argv)
{
    ParameterArgs par_args(argc, argv);

    string par_filename = par_args.key_exists("--parfile") ? par_args.get_string_parameter("--parfile")
                                                            : "../par/ray_transfer_disc_surface.par";
    ParameterFile par_file(par_filename.c_str());

    string out_filename = par_args.key_exists("--outfile") ? par_args.get_parameter<string>("--outfile")
                                                            : par_file.get_parameter<string>("outfile");
    string spec_filename = par_args.key_exists("--specfile") ? par_args.get_parameter<string>("--specfile")
                          : par_file.get_parameter<string>("specfile", out_filename.substr(0, out_filename.rfind('.')) + "_spectrum.csv");

    // image plane, as in ray_transfer_disc_wind.cpp
    const double dist = par_file.get_parameter<double>("dist");
    const double incl = (par_args.key_exists("--incl")) ? par_args.get_parameter<double>("--incl")
                                                         : par_file.get_parameter<double>("incl");
    const double spin = (par_args.key_exists("--spin")) ? par_args.get_parameter<double>("--spin")
                                                         : par_file.get_parameter<double>("spin");
    const double plane_phi0 = par_file.get_parameter<double>("plane_phi0", 0);
    // grid_type: "linear" (default) is ImagePlane's original uniformly-spaced (x0/xmax/Nx/y0/ymax/Ny) grid;
    // "log" is LogImagePlane's logarithmically-spaced grid (log_x_min/log_x_max/log_Nx/log_Nlinx and the
    // y-equivalents). See src/raytracer/log_imageplane.h and the pixel-grid query methods in imageplane.h.
    const string grid_type = par_file.get_parameter<string>("grid_type", "linear");
    const double symp_step = par_file.get_parameter<double>("symp_step", -1);
    const int symp_order = par_file.get_parameter<int>("symp_order", 6);
    const double max_tstep = par_file.get_parameter<double>("max_tstep", 0.01);

    // co-rotating gas layer above the disc surface (DiscSurface, rtfield.h)
    const double R_in = par_file.get_parameter<double>("R_in");
    const double R_out = par_file.get_parameter<double>("R_out");
    const double z_min = par_file.get_parameter<double>("z_min");
    const double z_max = par_file.get_parameter<double>("z_max");
    const string density_mode_str = par_file.get_parameter<string>("density_mode", "constant");
    const double n0 = par_file.get_parameter<double>("n0");
    const double density_index = par_file.get_parameter<double>("density_index", 0.0);

    DiscSurfaceDensityMode density_mode;
    if (density_mode_str == "radius_powerlaw") density_mode = DiscSurfaceDensityMode::RadiusPowerLaw;
    else if (density_mode_str == "height_powerlaw") density_mode = DiscSurfaceDensityMode::HeightPowerLaw;
    else density_mode = DiscSurfaceDensityMode::Constant;
    cout << "Gas layer density mode: " << density_mode_str << endl;

    // emitting annulus of the disc surface (DiscContinuumSource, continuum_source.h)
    const double R_in_corona = par_file.get_parameter<double>("R_in_corona");
    const double R_out_corona = par_file.get_parameter<double>("R_out_corona");
    const double I_corona = par_file.get_parameter<double>("I_corona", 1.0);

    // full opaque disc (inner edge: ISCO); the emitting annulus above is a subset of this range and takes
    // priority over it (see the corona/disc check reorder in RayTransfer::trace_pixel, ray_transfer.cpp)
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
    cout << "ISCO at " << r_isco << ", emitting annulus " << R_in_corona << " - " << R_out_corona << endl;

    // wind line-emission source function: "density" (density_scale * density) or "powerlaw" (a power law
    // in the wind point's own cylindrical radius, WindSourceMode::PowerLaw, ray_transfer.h) -- there is no
    // "illumination" option here: an accurate illumination() for an extended disc annulus would need
    // integrating over the whole source, so "powerlaw" is the recommended physically-motivated option for
    // this application instead (see ray_transfer.h's WindSourceMode comment).
    const string source_mode_str = par_file.get_parameter<string>("source_mode", "powerlaw");
    const WindSourceMode source_mode = (source_mode_str == "density") ? WindSourceMode::Density
                                                                       : WindSourceMode::PowerLaw;
    const double density_scale = par_file.get_parameter<double>("density_scale", 1.0);
    const double powerlaw_norm = par_file.get_parameter<double>("powerlaw_norm", 1.0);
    const double powerlaw_ref_r = par_file.get_parameter<double>("powerlaw_ref_r", R_in);
    const double powerlaw_index = par_file.get_parameter<double>("powerlaw_index", 0.0);
    cout << "Wind source function: " << source_mode_str << endl;

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
                                                     log_y_min, log_y_max, log_Ny, log_Nliny, spin, plane_phi0);
    }
    else
    {
        const double x0 = par_file.get_parameter<double>("x0");
        const double xmax = par_file.get_parameter<double>("xmax");
        const int Nx = par_file.get_parameter<int>("Nx");
        const double y0 = par_file.get_parameter<double>("y0", x0);
        const double ymax = par_file.get_parameter<double>("ymax", xmax);
        const int Ny = par_file.get_parameter<int>("Ny", Nx);
        const double dx = (xmax - x0) / Nx;
        const double dy = (ymax - y0) / Ny;

        plane = make_unique<ImagePlane<double>>(dist, incl, x0, xmax, dx, y0, ymax, dy, spin, plane_phi0);
    }

    DiscSurface<double> gas(R_in, R_out, z_min, z_max, density_mode, n0, density_index);
    DiscContinuumSource<double> corona(R_in_corona, R_out_corona, I_corona);
    DiscWithISCODestination<double> disc(r_isco, r_out_disc);
    LineTransition<double> line{line_energy, doppler_width, kappa0};
    SpectrumGrid<double> bins = SpectrumGrid<double>::linspace(energy_min, energy_max, n_energy);

    RayTransfer<double> rt(*plane, spin, gas, line, bins, &disc, &corona, source_mode, density_scale);
    rt.set_symplectic_step(symp_step);
    rt.set_symplectic_order(symp_order);
    rt.set_max_tstep(max_tstep);   // far-field cap on the coordinate-time step (default MAXDT)
    rt.set_powerlaw_source(powerlaw_norm, powerlaw_ref_r, powerlaw_index);

    const int Nx = rt.get_Nx(), Ny = rt.get_Ny();

    rt.run_raytrace();
    cout << rt.n_corona << " / " << (Nx * Ny) << " pixels hit the disc's emitting annulus" << endl;

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
        cout << "Gas layer optical depth: peak tau = " << tau_max << ", mean tau = " << (tau_sum / n_pix)
             << ", " << n_thick << " / " << n_pix << " sightlines optically thick (tau > 1)" << endl;
        cout << "Peak optical depth to the emitting annulus (absorption trough depth against the continuum): "
             << rt.tau_corona_max << endl;
    }

    ofstream spec_csv(spec_filename);
    spec_csv << "energy,line_flux,total_flux,residual" << endl;
    for (int j = 0; j < n_energy; j++)
        spec_csv << scientific << setprecision(8) << bins.energy[j] << ',' << rt.spec_line[j] << ','
                 << rt.spec_total[j] << ',' << (rt.continuum_total > 0 ? rt.spec_total[j] / rt.continuum_total : 0) << endl;
    spec_csv.close();
    cout << "Wrote " << spec_filename << endl;

    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("RayTransfer disc-surface gas layer continuum and line/continuum flux cube");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "ray_transfer_disc_surface");
    fits.write_keyword("DIST", "Distance to image plane", dist);
    fits.write_keyword("INCL", "Inclination of line of sight", incl);
    fits.write_keyword("SPIN", "Black hole spin", spin);
    fits.write_keyword("ISCO", "Innermost stable circular orbit", r_isco);
    fits.write_keyword("RDISCOUT", "Outer radius of disc", r_out_disc);
    fits.write_keyword("RCORIN", "Emitting annulus inner radius", R_in_corona);
    fits.write_keyword("RCOROUT", "Emitting annulus outer radius", R_out_corona);
    fits.write_keyword("RGASIN", "Gas layer inner cylindrical radius", R_in);
    fits.write_keyword("RGASOUT", "Gas layer outer cylindrical radius", R_out);
    fits.write_keyword("LINEEN", "Line rest energy", line_energy);
    fits.write_keyword("NCORONA", "Pixels that hit the disc's emitting annulus", rt.n_corona);
    fits.write_keyword("GRIDTYPE", "Image-plane pixel grid: linear or log", grid_type);
    if (grid_type == "log")
    {
        fits.write_keyword("LOGXMIN", "Log grid inner x edge", par_file.get_parameter<double>("log_x_min"));
        fits.write_keyword("LOGXMAX", "Log grid outer x edge", par_file.get_parameter<double>("log_x_max"));
        fits.write_keyword("LOGNX", "Log grid: log-spaced bins per side, x", par_file.get_parameter<int>("log_Nx"));
        fits.write_keyword("LOGNLINX", "Log grid: linear-zone bins, x", par_file.get_parameter<int>("log_Nlinx"));
        fits.write_keyword("LOGYMIN", "Log grid inner y edge", par_file.get_parameter<double>("log_y_min", par_file.get_parameter<double>("log_x_min")));
        fits.write_keyword("LOGYMAX", "Log grid outer y edge", par_file.get_parameter<double>("log_y_max", par_file.get_parameter<double>("log_x_max")));
        fits.write_keyword("LOGNY", "Log grid: log-spaced bins per side, y", par_file.get_parameter<int>("log_Ny", par_file.get_parameter<int>("log_Nx")));
        fits.write_keyword("LOGNLINY", "Log grid: linear-zone bins, y", par_file.get_parameter<int>("log_Nliny", par_file.get_parameter<int>("log_Nlinx")));
    }

    // Self-describing pixel-center coordinates and cell widths, valid for either grid_type -- see
    // ray_transfer_disc_wind.cpp's identical block for the rationale.
    {
        vector<double> xcol(Nx), dxcol(Nx);
        for (int ix = 0; ix < Nx; ix++) { xcol[ix] = plane->pixel_x(ix); dxcol[ix] = plane->pixel_dx(ix); }
        char* ttype[] = {(char*)"X", (char*)"DX"};
        char* tform[] = {(char*)COL_FLOAT64, (char*)COL_FLOAT64};
        char* tunit[] = {(char*)"Rg", (char*)"Rg"};
        fits.create_table("XGRID", 2, Nx, ttype, tform, tunit);
        fits.write_table_column(xcol.data(), Nx);
        fits.write_table_column(dxcol.data(), Nx);
        fits.write_comment("Pixel-center x coordinate and column width for each image-plane column");
    }
    {
        vector<double> ycol(Ny), dycol(Ny);
        for (int iy = 0; iy < Ny; iy++) { ycol[iy] = plane->pixel_y(iy); dycol[iy] = plane->pixel_dy(iy); }
        char* ttype[] = {(char*)"Y", (char*)"DY"};
        char* tform[] = {(char*)COL_FLOAT64, (char*)COL_FLOAT64};
        char* tunit[] = {(char*)"Rg", (char*)"Rg"};
        fits.create_table("YGRID", 2, Ny, ttype, tform, tunit);
        fits.write_table_column(ycol.data(), Ny);
        fits.write_table_column(dycol.data(), Ny);
        fits.write_comment("Pixel-center y coordinate and row width for each image-plane row");
    }

    fits.write_image(*rt.continuum_map, Nx, Ny, false);
    fits.set_ext_name("CONTINUUM");
    fits.write_comment("Unabsorbed disc continuum (0 where the ray did not reach the emitting annulus)");

    fits.write_image(*rt.tau_map, Nx, Ny, false);
    fits.set_ext_name("TAU");
    fits.write_comment("Peak gas-layer optical depth over the energy grid, per line of sight (max_j absorption[j])");

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
