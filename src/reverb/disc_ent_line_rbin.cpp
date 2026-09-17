/*
 * disc_ent_spectrum.cu
 *
 * Computes the time/energy grid (impulse response function) for X-ray reflection from a point source above the accretion disc
 * given an input reflection spectrum
 *
 * Computes the response of a single line first, then convolves with the reflection spectrum, rebinning as specified
 *
 * Creates a FITS image of lag time with energy from bottom to top and frequency from left to right
 *
 *  Created on: 10 Feb 2015
 *      Author: drw
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;

#include "raytracer/pointsource.h"

#include "include/hdf5_disc_transfer.h"
#include "include/array.h"
#include "include/fits_output.h"
#include "include/par_file.h"
#include "include/par_args.h"
#include "../include/disc.h"


int main(int argc, char **argv)
{
    ParameterArgs par_args(argc, argv);

    // parameter configuration file
    char default_par_filename[] = "../par/disc_ent_line_rbin.par";
    char *par_filename;
    if(par_args.key_exists("--parfile"))
    {
        string par_filename_str = par_args.get_string_parameter("--parfile");
        par_filename = (char *) par_filename_str.c_str();
    }
    else
        par_filename = default_par_filename;

    double source[4];

    double *en, *t, *disc_r, *area;

    long disc_count = 0;

    PointSource<double> *raytrace_source;

    ParameterFile par_file(par_filename);
    string out_filename = (par_args.key_exists("--outfile")) ? par_args.get_parameter<string>("--outfile")
                                                             : par_file.get_parameter<string>("outfile");
    string trf_filename = (par_args.key_exists("--trffile")) ? par_args.get_parameter<string>("--trffile")
                                                             : par_file.get_parameter<string>("trffile");
    par_file.get_parameter_array("source", source, 4);
    double V = par_file.get_parameter<double>("V");
    double spin = (par_args.key_exists("--spin")) ? par_args.get_parameter<double>("--spin") :
                         par_file.get_parameter<double>("spin");
    double mass = par_file.get_parameter<double>("mass", -1);
    double cosalpha0 = par_file.get_parameter<double>("cosalpha0", -0.9999);
    double cosalphamax = par_file.get_parameter<double>("cosalphamax", 0.9999);
    double dcosalpha = par_file.get_parameter<double>("dcosalpha");
    double beta0 = par_file.get_parameter<double>("beta0", -1 * M_PI);
    double betamax = par_file.get_parameter<double>("betamax", M_PI);
    double dbeta = par_file.get_parameter<double>("dbeta");
    double r_max = par_file.get_parameter<double>("rmax", 1000);
    double r_disc = par_file.get_parameter<double>("rdisc", 500);
    double r_min = par_file.get_parameter<double>("rmin", -1);
    int Nr = par_file.get_parameter<int>("Nr");
    bool logbin_r = par_file.get_parameter<bool>("logbin_r", true);
    double en0 = par_file.get_parameter<double>("en0");
    double enmax = par_file.get_parameter<double>("enmax");
    int Nen = par_file.get_parameter<int>("Nen");
    double line_en = par_file.get_parameter<double>("line_en", 6.4);
    bool logbin_en = par_file.get_parameter<bool>("logbin_en", false);
    double t_continuum = (par_args.key_exists("--t_continuum")) ? par_args.get_parameter<double>("--t_continuum") :
                         par_file.get_parameter<double>("t_continuum");
    double t0 = par_file.get_parameter<double>("t0", 0);
    double tmax = par_file.get_parameter<double>("tmax");
    int Nt = par_file.get_parameter<int>("Nt");
    double Gamma = par_file.get_parameter<double>("Gamma", 1);
    string integrator_str = par_file.get_parameter<string>("integrator", "rk45");
    double rk45_tol = par_file.get_parameter<double>("rk45_tol", 1e-8);
    double symp_step = par_file.get_parameter<double>("symp_step", -1);    // Mino-time step for the symplectic integrator (<= 0: 1/precision)
    int symp_order = par_file.get_parameter<int>("symp_order", 6);         // 2, 4 or 6
    double max_tstep = par_file.get_parameter<double>("max_tstep", MAXDT);
    int run_max = par_file.get_parameter<int>("run_max", -1);
    int show_progress = (par_args.key_exists("--show_progress")) ? par_args.get_parameter<int>("--show_progress")
	                                        : par_file.get_parameter<int>("show_progress", 1);

    Integrator integrator;
    if (integrator_str == "euler")
        integrator = Integrator::Euler;
    else if (integrator_str == "rk4")
        integrator = Integrator::RK4;
    else if (integrator_str == "symplectic")
        integrator = Integrator::Symplectic;
    else
        integrator = Integrator::RK45;

    if(par_args.key_exists("--source_h")) source[1] = par_args.get_parameter<double>("--source_h");

    double r_isco = kerr_isco(spin, +1);
    if(r_min < 0) r_min = r_isco;

    double dr = (logbin_r) ? exp(log(r_disc / r_min) / Nr) : (r_disc - r_min) / Nr;

    // conversion factor between natural units and seconds (mass in M_sun), or if mass < 0, just do the computation in natural units
    double rgs = (mass > 0) ? 6.67E-11 * mass * (2E30) / pow(3E8, 3) : 1.0;

    double dt = (tmax - t0) / (Nt - 1);
    double den = (logbin_en) ? exp(log(enmax / en0) / (Nen - 1)) : (enmax - en0) / (Nen - 1);

    cout << "Time bins:" << ' ' << t0 << ':' << dt << ':' << t0 + (Nt - 1) * dt << " (" << Nt << " bins)" << endl;
    cout << "Energy bins:" << ' ' << en0 << ':' << den << ':' << enmax << " (" << Nen << " "
         << ((logbin_en) ? "logarithmic" : "linear") << " bins)" << endl;

    Array2D<double> ent(Nen, Nt);

    en = new double[Nen];
    t = new double[Nt];
    for(int j = 0; j < Nen; j++)
    {
        en[j] = (logbin_en) ? en0 * pow(den, j) : en0 + j * den;
    }
    for(int k = 0; k < Nt; k++)
    {
        t[k] = t0 + k * dt;
    }

    disc_r = new double[Nr];
    area = new double[Nr];
    for(int ir=0; ir<Nr; ir++)
    {
        disc_r[ir] = (logbin_r) ? r_min * pow(dr, ir) : r_min + ir * dr;

        double annulus_dr = (logbin_r) ? disc_r[ir]*(dr - 1) : dr;
        area[ir] = integrate_disc_area(disc_r[ir], disc_r[ir] + annulus_dr, spin);
    }

    Array<int> disc_rays(Nr);
    Array<double> disc_flux(Nr);
    Array<double> disc_emis(Nr);
    Array<double> disc_redshift(Nr);
    Array<double> disc_time(Nr);

    const int nbeta = static_cast<int>((betamax - beta0) / dbeta);
    const int ncosalpha = static_cast<int>((cosalphamax - cosalpha0) / dcosalpha);
    const int total_rays = nbeta * ncosalpha;
    const int runs = (run_max > 0) ? (total_rays - 1) / run_max + 1 : 1;
    const double run_dcosalpha = (cosalphamax - cosalpha0) / runs;

    for(int run = 1; run <= runs; run++)
    {
        double run_cosalpha0 = cosalpha0 + (run - 1) * run_dcosalpha;
        double run_cosalphamax = ((run_cosalpha0 + run_dcosalpha) < cosalphamax) ? (run_cosalpha0 + run_dcosalpha)
                                                                                 : cosalphamax;

        cout << endl << "Run " << run << '/' << runs << endl;
        cout << "cosalpha = " << run_cosalpha0 << " to " << run_cosalphamax << endl;

        PointSource<double>* raytrace_source = new PointSource<double>(source, V, spin, TOL, dcosalpha, dbeta, cosalpha0, cosalphamax, beta0, betamax);

        raytrace_source->redshift_start();
        raytrace_source->run_raytrace(integrator, M_PI_2, r_max, show_progress);
        raytrace_source->range_phi();
        raytrace_source->redshift(-1.0, false);

        for(int ray = 0; ray < raytrace_source->get_count(); ray++)
        {
            if(raytrace_source->rays[ray].status > 0)
            {
                double cartx, carty, cartz;
                cartesian<double>(cartx, carty, cartz, raytrace_source->rays[ray].r, raytrace_source->rays[ray].theta,
                                  raytrace_source->rays[ray].phi, spin);
                if(cartz < 1E-2 && raytrace_source->rays[ray].r <= r_disc && raytrace_source->rays[ray].r >= r_min)
                {
                    int ir = (logbin_r) ? static_cast<int>( log(raytrace_source->rays[ray].r / r_min) / log(dr))
                                        : static_cast<int>((raytrace_source->rays[ray].r - r_min) / dr);

                    if(ir >= 0 && ir < Nr)
                    {
                        ++disc_rays[ir];

                        // primary flux to work out returning radiation normalisation
                        // fraction of rays from the primary source hitting this part of the disc
                        // multiplied by the redshift to obtain the photon arrival rate in the rest frame
                        disc_flux[ir] += 1 / raytrace_source->rays[ray].redshift;

                        // emissivity in the rest frame of the disc material
                        disc_emis[ir] += 1 / pow(raytrace_source->rays[ray].redshift, Gamma);

                        disc_redshift[ir] += raytrace_source->rays[ray].redshift;
                        disc_time[ir] += raytrace_source->rays[ray].t;
                    }

                    ++disc_count;
                }
            }
        }

        delete raytrace_source;
    }

    cout << endl << disc_count << " photons hit the disc" << endl;

    disc_redshift /= disc_rays;
    disc_time /= disc_rays;

    // get the transfer function from disc to observer
    DiscTransfer<double> trf((char *) trf_filename.c_str());

    cout << "Computing energy/time grid..." << endl;

    int ir;
    double r;
    for(ir = 0, r = r_min; ir < Nr; ir++, r = (logbin_r) ? r * dr : r * dr)
    {
        if(disc_rays[ir] == 0) continue;

        const int trf_irmin = trf.r_index(r);
        const int trf_irmax = trf.r_index((logbin_r) ? r * dr : r + dr);

        for(int trf_ir = trf_irmin; trf_ir < trf_irmax; trf_ir++)
        {
            for(int trf_iphi = 0; trf_iphi < trf.Nphi; trf_iphi++)
            {
                const int trf_idx = trf.array_index(trf_ir, trf_iphi);

                const double energy = line_en / trf.g[trf_idx];
                const double t_obs = trf.t[trf_idx];

                double t_tot = t_obs + disc_time[ir];

                // the continuum arrives at t=0
                t_tot -= t_continuum;

                // after subtracting the continuum, convert the time lag into seconds
                t_tot *= rgs;

                const int it = static_cast<int>((t_tot - t0) / dt);

                const int ien = (logbin_en) ? static_cast<int>( log(energy / en0) / log(den))
                                            : static_cast<int>((energy - en0) / den);

                if(it >= 0 && it < Nt && ien >= 0 && ien < Nen)
                {
                    // note we need to divide by the number of rbins to make sure that each annulus
                    // is weighted appropriately
                    ent[ien][it] += disc_emis[ir] / (pow(trf.g[trf_idx], 3) * (trf_irmax - trf_irmin));
                }
            }
        }
    }


    FITSOutput<double> fits(out_filename);
    fits.create_primary();
    fits.write_comment("Accretion disc line response");
    fits.write_keyword("GENERATOR", "Simulation results were generated by this software", "disc_ent_line");
    fits.write_keyword("RAYTRACESOURCE", "Raytrace source configuration", RAYTRACE_SOURCE);
    fits.write_keyword("T", "Ray time at source", source[0]);
    fits.write_keyword("R", "Source r co-ordinate", source[1]);
    fits.write_keyword("THETA", "Source theta co-ordinate", source[2]);
    fits.write_keyword("PHI", "Source phi co-ordinate", source[3]);
    fits.write_keyword("V", "Source angular velocity d(phi)/dt", V);
    fits.write_keyword("DCOSALPH", "Step in cos source polar angle", dcosalpha);
    fits.write_keyword("DBETA", "Step in source azimuthal angle", dbeta);
    fits.write_keyword("NRAYS", "Number of rays", static_cast<int>((cosalphamax - cosalpha0) / dcosalpha) *
                                                  static_cast<int>((betamax - beta0) / dbeta));
    fits.write_keyword("DISCRAYS", "Rays hitting disc", disc_count);
    fits.write_keyword("SPIN", "Black hole spin", spin);
    fits.write_keyword("MASS", "Black hole mass", mass);
    fits.write_keyword("RGS", "Gravitational units to seconds conversion (GM/c^3)", rgs);
    fits.write_keyword("ISCO", "Innermost stable circular orbit", r_isco);
    fits.write_keyword("RMAX", "Raytrace max radius", r_max);
    fits.write_keyword("LINEEN", "Rest frame energy of emission line", line_en);
    fits.write_keyword("TRF", "Disc to observer transfer function used", trf_filename);

    fits.write_image(ent, Nen, Nt, true);
    fits.set_ext_name((char *) "RESPONSE");
    fits.write_comment("Total energy/time grid for disc reflection");
    fits.write_keyword("AXIS1", "Quantity along X axis of image", "TIME");
    fits.write_keyword("AXIS2", "Quantity along Y axis of image", "ENERGY");
    fits.write_keyword("PIXVAL", "Pixel value quantity", "PHOTON RATE");
    fits.write_keyword("PIXUNIT", "Pixel value unit", "PH/DT");
    fits.write_keyword("T0", "Start of time axis", t0);
    fits.write_keyword("DT", "Time step", dt);
    fits.write_keyword("NT", "Number of time steps", Nt);
    if(mass > 0)
        fits.write_keyword("TUNIT", "Units of time axis", "SECOND");
    else
        fits.write_keyword("TUNIT", "Units of time axis", "GM/c^3");
    fits.write_keyword("TSTART", "Start time relative to flash (continuum arrival)", t_continuum);
    fits.write_keyword("EN0", "Start of energy axis", en0);
    fits.write_keyword("ENMAX", "End of energy axis", enmax);
    fits.write_keyword("NEN", "Number of energy steps", Nen);
    fits.write_keyword("DEN", "Energy step", den);
    fits.write_keyword("ENLOG", "Energy axis is logarithmic", logbin_en);
    fits.write_keyword("ENUNIT", "Units of energy axis", "REDSHIFT");

    // -- write a table describing the disc--

    char *ttype[] = {"r", "area", "count" , "flux", "emis", "time", "redshift"};
    char *tform[] = {COL_FLOAT, COL_FLOAT, COL_INT32, COL_FLOAT, COL_FLOAT, COL_FLOAT, COL_FLOAT};
    char *tunit[] = {"\0", "\0", "\0", "\0", "\0", "\0", "\0"};

    fits.create_table("DISC", 7, Nr, ttype, tform, tunit);
    fits.write_table_column(disc_r, Nr);
    fits.write_table_column((double*)area, Nr);
    fits.write_table_column((int*)disc_rays, Nr);
    fits.write_table_column((double*)disc_flux, Nr);
    fits.write_table_column((double*)disc_emis, Nr);
    fits.write_table_column((double*)disc_time, Nr);
    fits.write_table_column((double*)disc_redshift, Nr);

    fits.close();

    cout << "Done" << endl;

    return 0;
}
