//
// imageplane_disc_image_rd.cpp
//
// Ray-traced disc image using the RayDestination API for configurable
// disc geometry and velocity field.
//
// Rays are propagated using the RK4 integrator and stopped when they
// reach the disc surface as defined by FlatDiscDestination. The redshift
// is computed using the full 4-velocity from the RayDestination object.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;

#include "../raytracer/imageplane.h"
#include "../raytracer/ray_destination.h"

#include "../include/fits_output.h"
#include "../include/array.h"
#include "../include/par_file.h"
#include "../include/par_args.h"

double powerlaw3(double r, double q1, double rb1, double q2, double rb2, double q3)
{
    if(r < rb1)
        return pow(r, -1 * q1);
    else if(r < rb2)
        return pow(rb1, q2 - q1) * pow(r, -1 * q2);
    else
        return pow(rb1, q2 - q1) * pow(rb2, q3 - q2) * pow(r, -1 * q3);
}

int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    // parameter configuration file
    char default_par_filename[] = "../par/imageplane_disc_image_rd.par";
    char *par_filename;
    if(par_args.key_exists("--parfile"))
    {
        string par_filename_str = par_args.get_string_parameter("--parfile");
        par_filename = (char *) par_filename_str.c_str();
    }
    else
        par_filename = default_par_filename;

    ParameterFile par_file(par_filename);
    string out_filename = (par_args.key_exists("--outfile")) ? par_args.get_parameter<string>("--outfile")
                                                             : par_file.get_parameter<string>("outfile");
    double dist = par_file.get_parameter<double>("dist");
    double incl = (par_args.key_exists("--incl")) ? par_args.get_parameter<double>("--incl")
                                                  : par_file.get_parameter<double>("incl");
    double plane_phi0 = par_file.get_parameter<double>("plane_phi0", 0);
    double spin = (par_args.key_exists("--spin")) ? par_args.get_parameter<double>("--spin")
                                                  : par_file.get_parameter<double>("spin");
    double r_disc = par_file.get_parameter<double>("r_disc");
    double theta_lim = par_file.get_parameter<double>("theta_lim", M_PI_2);
    double x0 = par_file.get_parameter<double>("x0", -1 * r_disc);
    double xmax = par_file.get_parameter<double>("xmax", r_disc);
    int Nx = par_file.get_parameter<int>("Nx");
    double y0 = par_file.get_parameter<double>("y0", x0);
    double ymax = par_file.get_parameter<double>("ymax", xmax);
    int Ny = par_file.get_parameter<int>("Ny", Nx);
    int img_Nx = par_file.get_parameter<int>("img_Nx", Nx);
    int img_Ny = par_file.get_parameter<int>("img_Ny", img_Nx);
    double q1 = par_file.get_parameter<double>("q1", 3);
    double rb1 = par_file.get_parameter<double>("rb1", 4);
    double q2 = par_file.get_parameter<double>("q2", 3);
    double rb2 = par_file.get_parameter<double>("rb2", 10);
    double q3 = par_file.get_parameter<double>("q3", 3);
    double precision = par_file.get_parameter<double>("precision", PRECISION);
    // ImagePlane now places the ray from pixel (x, y) on the (x, y) side of the black hole (y > 0 towards the
    // pole), so the image is correctly oriented without flipping; flip_image = 1 restores the old (inverted) output
    bool flip_image = par_file.get_parameter<bool>("flip_image", false);

    double dx = (xmax - x0) / Nx;
    double dy = (ymax - y0) / Ny;

    double img_dx = (xmax - x0) / img_Nx;
    double img_dy = (ymax - y0) / img_Ny;

    long disc_count = 0;

    Array2D<double> disc_flux(img_Nx, img_Ny);
    Array2D<double> disc_r(img_Nx, img_Ny);
    Array2D<double> disc_phi(img_Nx, img_Ny);
    Array2D<double> disc_enshift(img_Nx, img_Ny);
    Array2D<double> disc_time(img_Nx, img_Ny);
    Array2D<double> disc_emis(img_Nx, img_Ny);
    Array2D<int> disc_Nrays(img_Nx, img_Ny);

    disc_flux.zero();
    disc_r.zero();
    disc_phi.zero();
    disc_enshift.zero();
    disc_time.zero();
    disc_emis.zero();
    disc_Nrays.zero();

    double r_isco = kerr_isco<double>(spin, +1);
    cout << "ISCO at " << r_isco << endl;

    FlatDiscDestination<double> disc_dest(theta_lim);

    ImagePlane<double> raytrace_source(dist, incl, x0, xmax, dx, y0, ymax, dy, spin, plane_phi0,
                                       precision);

    raytrace_source.redshift_start();
    raytrace_source.run_raytrace(&disc_dest, Integrator::RK4, 1.1 * dist);
    raytrace_source.redshift(&disc_dest);
    raytrace_source.range_phi();

    for(int ray = 0; ray < raytrace_source.get_count(); ray++)
    {
        if(raytrace_source.rays[ray].steps > 0 &&
           raytrace_source.rays[ray].theta >= theta_lim - 1e-3 &&
           raytrace_source.rays[ray].r >= r_isco &&
           raytrace_source.rays[ray].r < r_disc &&
           raytrace_source.rays[ray].redshift > 0)
        {
            double x = raytrace_source.rays[ray].alpha;
            double y = raytrace_source.rays[ray].beta;
            int ix = static_cast<int>((x - x0) / img_dx);
            int iy = static_cast<int>((y - y0) / img_dy);

            if(flip_image)
                iy = img_Ny - iy - 1;

            if(ix >= 0 && ix < img_Nx && iy >= 0 && iy < img_Ny)
            {
                ++disc_Nrays[ix][iy];

                double emis = powerlaw3(raytrace_source.rays[ray].r, q1, rb1, q2, rb2, q3);

                disc_flux[ix][iy] += emis / pow(raytrace_source.rays[ray].redshift, 3);
                disc_r[ix][iy] += raytrace_source.rays[ray].r;
                disc_phi[ix][iy] += raytrace_source.rays[ray].phi;
                disc_enshift[ix][iy] += 1. / raytrace_source.rays[ray].redshift;
                disc_time[ix][iy] += raytrace_source.rays[ray].t;
                disc_emis[ix][iy] += emis;

                ++disc_count;
            }
        }
    }

    cout << disc_count << " rays hit the disc" << endl;

    for(int ix = 0; ix < img_Nx; ix++)
        for(int iy = 0; iy < img_Ny; iy++)
            if(disc_Nrays[ix][iy] > 0)
                disc_flux[ix][iy] /= disc_Nrays[ix][iy];

    disc_r /= disc_Nrays;
    disc_phi /= disc_Nrays;
    disc_enshift /= disc_Nrays;
    disc_time /= disc_Nrays;
    disc_emis /= disc_Nrays;

    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Raytraced images of accretion disc (RayDestination API)");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "imageplane_disc_image_rd");
    fits.write_keyword("raytrace_source", "Raytrace source configuration", "imageplane");
    fits.write_keyword("DIST", "Distance to image plane", dist);
    fits.write_keyword("INCL", "Inclination of line of sight", incl);
    fits.write_keyword("SPIN", "Black hole spin", spin);
    fits.write_keyword("ISCO", "Innermost stable circular orbit", r_isco);
    fits.write_keyword("RDISC", "Maximum radius of disc", r_disc);
    fits.write_keyword("THLIM", "Disc surface polar angle", theta_lim);
    fits.write_keyword("Q1", "Emissivity profile inner power law index", q1);
    fits.write_keyword("RB1", "Emissivity profile inner break radius", rb1);
    fits.write_keyword("Q2", "Emissivity profile middle power law index", q2);
    fits.write_keyword("RB2", "Emissivity profile outer break radius", rb2);
    fits.write_keyword("Q3", "Emissivity profile outer power law index", q3);
    fits.write_keyword("NRAYS", "Number of rays", Nx * Ny);
    fits.write_keyword("DISCRAYS", "Rays hitting disc", disc_count);

    fits.write_image(disc_flux, img_Nx, img_Ny, false);
    fits.set_ext_name("FLUX");
    fits.write_comment("Observed flux");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "Image plane X");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "Image plane Y");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "FLUX");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "PH*EN^2/DT");
    fits.write_keyword("X0", "Start of X axis", x0);
    fits.write_keyword("XMAX", "End of X axis", xmax);
    fits.write_keyword("DX", "X step", dx);
    fits.write_keyword("NX", "Number of pixels in X", img_Nx);
    fits.write_keyword("XUNIT", "Units of X axis", "RG");
    fits.write_keyword("Y0", "Start of Y axis", y0);
    fits.write_keyword("YMAX", "End of Y axis", ymax);
    fits.write_keyword("DY", "Y step", dy);
    fits.write_keyword("NY", "Number of pixels in Y", img_Ny);
    fits.write_keyword("YUNIT", "Units of Y axis", "RG");

    fits.write_image(disc_r, img_Nx, img_Ny, false);
    fits.set_ext_name("RADIUS");
    fits.write_comment("Emission radius");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "Image plane X");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "Image plane Y");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "RADIUS");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "RG");
    fits.write_keyword("X0", "Start of X axis", x0);
    fits.write_keyword("XMAX", "End of X axis", xmax);
    fits.write_keyword("DX", "X step", dx);
    fits.write_keyword("NX", "Number of pixels in X", img_Nx);
    fits.write_keyword("XUNIT", "Units of X axis", "RG");
    fits.write_keyword("Y0", "Start of Y axis", y0);
    fits.write_keyword("YMAX", "End of Y axis", ymax);
    fits.write_keyword("DY", "Y step", dy);
    fits.write_keyword("NY", "Number of pixels in Y", img_Ny);
    fits.write_keyword("YUNIT", "Units of Y axis", "RG");

    fits.write_image(disc_phi, img_Nx, img_Ny, false);
    fits.set_ext_name("PHI");
    fits.write_comment("Phi co-ordinate of emission");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "Image plane X");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "Image plane Y");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "PHI");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "RAD");
    fits.write_keyword("X0", "Start of X axis", x0);
    fits.write_keyword("XMAX", "End of X axis", xmax);
    fits.write_keyword("DX", "X step", dx);
    fits.write_keyword("NX", "Number of pixels in X", img_Nx);
    fits.write_keyword("XUNIT", "Units of X axis", "RG");
    fits.write_keyword("Y0", "Start of Y axis", y0);
    fits.write_keyword("YMAX", "End of Y axis", ymax);
    fits.write_keyword("DY", "Y step", dy);
    fits.write_keyword("NY", "Number of pixels in Y", img_Ny);
    fits.write_keyword("YUNIT", "Units of Y axis", "RG");

    fits.write_image(disc_enshift, img_Nx, img_Ny, false);
    fits.set_ext_name("ENSHIFT");
    fits.write_comment("Observed photon energy");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "Image plane X");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "Image plane Y");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "ENERGY");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "OBS/EMIT");
    fits.write_keyword("X0", "Start of X axis", x0);
    fits.write_keyword("XMAX", "End of X axis", xmax);
    fits.write_keyword("DX", "X step", dx);
    fits.write_keyword("NX", "Number of pixels in X", img_Nx);
    fits.write_keyword("XUNIT", "Units of X axis", "RG");
    fits.write_keyword("Y0", "Start of Y axis", y0);
    fits.write_keyword("YMAX", "End of Y axis", ymax);
    fits.write_keyword("DY", "Y step", dy);
    fits.write_keyword("NY", "Number of pixels in Y", img_Ny);
    fits.write_keyword("YUNIT", "Units of Y axis", "RG");

    fits.write_image(disc_time, img_Nx, img_Ny, false);
    fits.set_ext_name("TIME");
    fits.write_comment("Ray travel time to image plane");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "Image plane X");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "Image plane Y");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "TIME");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "GM/c^3");
    fits.write_keyword("X0", "Start of X axis", x0);
    fits.write_keyword("XMAX", "End of X axis", xmax);
    fits.write_keyword("DX", "X step", dx);
    fits.write_keyword("NX", "Number of pixels in X", img_Nx);
    fits.write_keyword("XUNIT", "Units of X axis", "RG");
    fits.write_keyword("Y0", "Start of Y axis", y0);
    fits.write_keyword("YMAX", "End of Y axis", ymax);
    fits.write_keyword("DY", "Y step", dy);
    fits.write_keyword("NY", "Number of pixels in Y", img_Ny);
    fits.write_keyword("YUNIT", "Units of Y axis", "RG");

    fits.write_image(disc_emis, img_Nx, img_Ny, false);
    fits.set_ext_name("EMIS");
    fits.write_comment("Accretion disc emissivity");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "Image plane X");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "Image plane Y");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "EMISSIVITY");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "PH/RG^2/DT");
    fits.write_keyword("X0", "Start of X axis", x0);
    fits.write_keyword("XMAX", "End of X axis", xmax);
    fits.write_keyword("DX", "X step", dx);
    fits.write_keyword("NX", "Number of pixels in X", img_Nx);
    fits.write_keyword("XUNIT", "Units of X axis", "RG");
    fits.write_keyword("Y0", "Start of Y axis", y0);
    fits.write_keyword("YMAX", "End of Y axis", ymax);
    fits.write_keyword("DY", "Y step", dy);
    fits.write_keyword("NY", "Number of pixels in Y", img_Ny);
    fits.write_keyword("YUNIT", "Units of Y axis", "RG");

    fits.close();

    cout << "Done" << endl;

    return 0;
}
