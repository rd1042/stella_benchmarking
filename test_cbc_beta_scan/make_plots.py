""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_beta_scan_plots, make_comparison_plots, plot_gmvus, plot_gzvs
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
from extract_sim_data import find_bpar_phi_ratio, find_apar_phi_ratio
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DIR = "./images/"
### Give the names of all the sims here - avoids needing to type them out
### in the methods.

## fapar = 0
# stella
sim_st_b00001_fapar0 = "stella_beta0.00001_fapar0/input"
sim_st_b001_fapar0 = "stella_beta0.001_fapar0/input"
sim_st_b002_fapar0 = "stella_beta0.002_fapar0/input"
sim_st_b01_fapar0  = "stella_beta0.010_fapar0/input"
# gs2
sim_gs2_b00001_fapar0 = "gs2_beta_scan_fapar0/_0.00001"
sim_gs2_b001_fapar0 = "gs2_beta_scan_fapar0/_0.0010"
sim_gs2_b002_fapar0 = "gs2_beta_scan_fapar0/_0.0020"
sim_gs2_b01_fapar0 = "gs2_beta_scan_fapar0/_0.0100"

## fbpar = 0
# stella
sim_st_b00001_fbpar0 = "stella_beta0.00001_fbpar0/input"
sim_st_b001_fbpar0 = "stella_beta0.001_fbpar0/input"
sim_st_b001_fbpar0_no_drive = "stella_beta0.001_fbpar0/input_zero_drive"
sim_st_b001_fbpar0_no_mag_drift = "stella_beta0.001_fbpar0/input_no_drifts"
sim_st_b001_fbpar0_no_mirror = "stella_beta0.001_fbpar0/input_no_mirror"
sim_st_b001_fbpar0_equal_masses = "stella_beta0.001_fbpar0_equal_masses/input"
sim_st_b002_fbpar0 = "stella_beta0.002_fbpar0/input"
sim_st_b003_fbpar0 = "stella_beta0.003_fbpar0/input"
sim_st_b004_fbpar0 = "stella_beta0.004_fbpar0/input"
sim_st_b005_fbpar0 = "stella_beta0.005_fbpar0/input"
sim_st_b01_fbpar0 = "stella_beta0.010_fbpar0/input"
# gs2
sim_gs2_b00001_fbpar0 = "gs2_beta_scan_fbpar0/_0.00001"
sim_gs2_b001_fbpar0 = "gs2_beta_scan_fbpar0/_0.0010"
sim_gs2_b002_fbpar0 = "gs2_beta_scan_fbpar0/_0.0020"
sim_gs2_b003_fbpar0 = "gs2_beta_scan_fbpar0/_0.0030"
sim_gs2_b004_fbpar0 = "gs2_beta_scan_fbpar0/_0.0040"
sim_gs2_b005_fbpar0 = "gs2_beta_scan_fbpar0/_0.0050"
sim_gs2_b01_fbpar0 = "gs2_beta_scan_fbpar0/_0.0100"

## fapar=1, fbpar=1
sim_st_b0 = "stella_beta0.000/input"
sim_st_b005 = "stella_beta0.005/input"
sim_st_b01 = "stella_beta0.010/input"
sim_st_b015 = "stella_beta0.015/input"
sim_st_b02 = "stella_beta0.020/input"
sim_st_b025 = "stella_beta0.025/input"
sim_st_b03 = "stella_beta0.030/input"

## The pickled files summarising G22 beta scans
pickle_gs2 = "gs2_beta_scan/omega_values.pickle"
pickle_gs2_fbpar0 = "gs2_beta_scan_fbpar0/omega_values.pickle"

def analyse_fbpar0_beta0001_results():
    """Compare sims, all with fbpar=0, fapar=1, beta=0.001, for which
    we try turning on and off different knobs."""

    make_comparison_plots([
                           sim_st_b001_fbpar0,
                           sim_st_b001_fbpar0_no_drive,
                           sim_st_b001_fbpar0_no_mag_drift,
                           sim_st_b001_fbpar0_no_mirror,
                           sim_st_b001_fbpar0_equal_masses,
                           sim_gs2_b001_fbpar0
                           ],
                          [
                           "stella",
                           "stella, zero gradients",
                           "stella, no magnetic drifts",
                           "stella, no mirror term",
                           "stella, m_e=1",
                           "GS2"
                           ],
                          "./termsoff_beta_0.001_fbpar0",
                          sim_types=[
                                     "stella",
                                     "stella",
                                     "stella",
                                     "stella",
                                     "stella",
                                     "gs2"
                                     ],
                           plot_apar=True,
                           plot_format=".eps"
                           )
    return


def analyse_fbpar0_results():
    """Compare omega(t) and phi(z), apar(z)
    between GS2 and stella results """

    print("Hello world")
    beta_strs = [

                 ]
    stella_sim_longnames = [

                            ]
    gs2_sim_longnames = [

                            ]
    for beta_idx in range(0, len(beta_strs)):
        stella_sim_longname = stella_sim_longnames[beta_idx]
        gs2_sim_longname = gs2_sim_longnames[beta_idx]
        beta_str = beta_strs[beta_idx]
        make_comparison_plots([
                               stella_sim_longname,
                               gs2_sim_longname,
                               ],
                              [
                               "stella",
                               "GS2",
                               ],
                              "./beta_" + beta_str + "_fbpar0",
                              sim_types=[
                                         "stella",
                                         "gs2",
                                         ],
                               plot_apar=True,
                               )


    return

def analyse_fapar0_results():
    """Compare omega(t) and phi(z), apar(z)
    between GS2 and stella results """

    print("Hello world")
    beta_strs = [
                 "0.00001",
                 "0.001",
                 "0.002",
                 #"0.010",
                 ]
    stella_sim_longnames = [

                            ]
    gs2_sim_longnames = [

                            ]
    for beta_idx in range(0, len(beta_strs)):
        stella_sim_longname = stella_sim_longnames[beta_idx]
        gs2_sim_longname = gs2_sim_longnames[beta_idx]
        beta_str = beta_strs[beta_idx]
        make_comparison_plots([
                               stella_sim_longname,
                               gs2_sim_longname,
                               ],
                              [
                               "stella",
                               "GS2",
                               ],
                              "./beta_" + beta_str + "_fapar0",
                              sim_types=[
                                         "stella",
                                         "gs2",
                                         ],
                               plot_bpar=True,
                               plot_format=".png"
                               )
        # plot_gmvus("stella_beta0.001_fapar0/input.out.nc")
        print("stella_sim_longname = ", stella_sim_longname)
        print("gs2_sim_longname = ", gs2_sim_longname)
        gs2_bpar_ratio = find_bpar_phi_ratio(gs2_sim_longname, "gs2")
        stella_bpar_ratio = find_bpar_phi_ratio(stella_sim_longname, "stella")
        print("gs2_bpar_ratio = ", gs2_bpar_ratio)
        print("stella_bpar_ratio2 = ", stella_bpar_ratio)

    return

def analyse_fapar0_changing_vpares():
    """Compare omega(t) and phi(z), apar(z)
    between GS2 and stella results """

    print("Hello world")

    stella_sim1_longname = "stella_beta0.010_fapar0/input"
    stella_sim2_longname = "stella_beta0.010_fapar0_higher_vpa/input"

    gs2_sim1_longname = "gs2_beta_scan_fapar0/_0.0100"

    # make_comparison_plots([
    #                        stella_sim1_longname,
    #                        stella_sim2_longname,
    #                        gs2_sim1_longname
    #                        ],
    #                       [
    #                        "stella",
    #                        "stella, higher vpa",
    #                        "GS2",
    #                        ],
    #                       "./beta_0.01_fapar0_varying_vpares",
    #                       sim_types=[
    #                                  "stella",
    #                                  "stella",
    #                                  "gs2",
    #                                  ],
    #                        plot_bpar=True,
    #                        )
    #
    gs2_bpar_ratio = find_bpar_phi_ratio(gs2_sim1_longname, "gs2")
    stella_bpar_ratio1 = find_bpar_phi_ratio(stella_sim1_longname, "stella")
    stella_bpar_ratio2 = find_bpar_phi_ratio(stella_sim2_longname, "stella")
    print("gs2_bpar_ratio = ", gs2_bpar_ratio)
    print("stella_bpar_ratio1 = ", stella_bpar_ratio1)
    print("stella_bpar_ratio2 = ", stella_bpar_ratio2)
    # plot_gmvus("stella_beta0.001_fapar0/input.out.nc", which="gvpa")
    # plot_gmvus("stella_beta0.010_fapar0_higher_vpa/input.out.nc", which="gvpa")
    return


def plot_geometry():
    """ """
    stella_outnc_longname = "stella_beta0.001_fbpar0/input.out.nc"
    gs2_outnc_longname = "gs2_beta_scan_fbpar0/_0.0010.out.nc"
    #view_ncdf_variables(stella_outnc_longname)
    #view_ncdf_variables(gs2_outnc_longname)
    plot_gmvus(stella_outnc_longname)

    sys.exit()

    time, theta, gs2_energy, gs2_lambda = extract_data_from_ncdf(gs2_outnc_longname,
                                     't', 'theta', 'energy', 'lambda')
    print("len(time), len(theta), len(energy), len(lambda) = ", len(time), len(theta), len(gs2_energy), len(gs2_lambda))
    gs2_g = np.loadtxt("gs2_beta_scan_fbpar0/_0.0100.dist", skiprows=1)
    print("gs2_g.shape = ", gs2_g.shape)
    # GS2's g is in a set of (n(energy)*n(lambda)) x 8 blocks.
    # Each row is vpa, vpe, energy(ie,is), al(il), xpts(ie,is), ypts(il), real(gtmp(1)), real(gtmp(2))
    # The number of blocks is nstep/(nwrite*nwrite_mul)
    block_size = len(gs2_energy) * len(gs2_lambda)
    nblocks = len(gs2_g)/block_size
    final_step_g = gs2_g[-block_size:, :]
    print("len(final_step_g), block_size = ", len(final_step_g), block_size)

    ## Code to plot g for GS2
    # gvmus = gvmus[-1]   # spec, mu, vpa
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # counter=0
    #
    # for mu_idx in range(0, len(mu)):
    #     counter += 1
    #     g_ion_vpa = gvmus[0, mu_idx, :]
    #     g_electron_vpa = gvmus[1, mu_idx, :]
    #     ax1.plot(vpa, g_ion_vpa)
    #     ax2.plot(vpa, g_electron_vpa)
    #
    #     if counter == 5:
    #         plt.show()
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot(211)
    #         ax2 = fig.add_subplot(212)
    #         counter=0
    #
    # plt.show()

    ## Code to compare geometry between stella and gs2
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # ax1.plot(z, gds2)
    # ax1.plot(z, gds21)
    # ax1.plot(z, gds22)
    # ax2.plot(z, bmag)
    # ax2.plot(z, gradpar)
    #
    # theta, gds2, gds21, gds22, bmag, gradpar = extract_data_from_ncdf(gs2_outnc_longname,
    #                                 'theta', 'gds2', 'gds21', 'gds22', 'bmag', 'gradpar')
    # ax1.plot(theta, gds2, linestyle="-.")
    # ax1.plot(theta, gds21, linestyle="-.")
    # ax1.plot(theta, gds22, linestyle="-.")
    # ax2.plot(theta, bmag, linestyle="-.")
    # ax2.plot(theta, gradpar, linestyle="-.")
    #
    #
    # plt.show()

    return


def plot_beta_scans():
    """ """
    stella_sim_longnames = [

                            ]
    stella_beta_vals = [
                        0.0,
                        0.005,
                        0.01,
                        0.015,
                        0.02,
                        0.025,
                        0.03
                        ]


    make_beta_scan_plots(stella_sim_longnames,
                         stella_beta_vals,
                          "./test_cbc_beta_scan",
                          gs2_pickle=gs2_pickle
                         )

    stella_sim_longnames = [

                            ]
    stella_beta_vals = [
                        0.0,
                        0.001,
                        0.002,
                        0.003,
                        0.004,
                        0.005,
                        ]



    make_beta_scan_plots(stella_sim_longnames,
                         stella_beta_vals,
                          "./test_cbc_beta_scan_fbpar0",
                         gs2_pickle=gs2_pickle
                         )
    return

def plot_gvmus_for_fbpar0():
    """Take a look at the distribrution function for
    fbpar=0 sims """

    stella_sim_longname = "stella_beta0.010_fbpar0/input"
    stella_sim_longname_higher_vpa = "stella_beta0.010_fbpar0_higher_vpa/input"
    stella_outnc_longname = stella_sim_longname + ".out.nc"
    stella_outnc_longname_higher_vpa = stella_sim_longname_higher_vpa + ".out.nc"
    make_comparison_plots([
                stella_sim_longname,
                stella_sim_longname_higher_vpa,
                ],
                [
                "nvgrid=36",
                "nvgrid=108",
                ],
                "./beta0.01_fbpar0_vpa_res_test",
                sim_types=[
                "stella",
                "stella",
                ],
                plot_apar=True,
                )
    plot_gmvus(stella_outnc_longname)
    plot_gmvus(stella_outnc_longname_higher_vpa)


    return

def plot_gzvs_for_fbpar0():
    """Take a look at the distribrution function for
    fbpar=0 sims """

    stella_sim_longname = "stella_beta0.010_fbpar0/input"
    stella_outnc_longname = stella_sim_longname + ".out.nc"
    plot_gzvs(stella_outnc_longname)


    return


def make_comparison_plots_many(stella_sim_longnames, gs2_sim_longnames,
                                beta_strs, prefix, plot_apar = False,
                                plot_bpar = False, plot_format=".png"):
    """ """

    for beta_idx in range(0, len(beta_strs)):
        stella_sim_longname = stella_sim_longnames[beta_idx]
        gs2_sim_longname = gs2_sim_longnames[beta_idx]
        beta_str = beta_strs[beta_idx]
        make_comparison_plots([
                               stella_sim_longname,
                               gs2_sim_longname,
                               ],
                              [
                               "stella",
                               "GS2",
                               ],
                              IMAGE_DIR + prefix + "beta_" + beta_str,
                              sim_types=[
                                         "stella",
                                         "gs2",
                                         ],
                               plot_apar=plot_apar,
                               plot_bpar=plot_bpar,
                               plot_format=plot_format
                               )

def plot_fapar_fbpar_on():
    """ """
    ## Beta scan
    stella_sim_longnames = [
                            sim_st_b0,
                            sim_st_b005,
                            sim_st_b01,
                            sim_st_b015,
                            sim_st_b02,
                            sim_st_b025,
                            sim_st_b03
                            ]
    stella_beta_vals = [
                        0.,
                        0.005,
                        0.01,
                        0.015,
                        0.02,
                        0.025,
                        0.03,
                        ]
    make_beta_scan_plots(stella_sim_longnames,
                         stella_beta_vals,
                         IMAGE_DIR + "test_cbc_beta_scan",
                         gs2_pickle=pickle_gs2)

    return

def plot_fapar0():
    """ """
    stella_sim_longnames = [
                            sim_st_b00001_fapar0,
                            sim_st_b001_fapar0,
                            sim_st_b002_fapar0,
                            sim_st_b01_fapar0,
                            ]
    gs2_sim_longnames = [
                         sim_gs2_b00001_fapar0,
                         sim_gs2_b001_fapar0,
                         sim_gs2_b002_fapar0,
                         sim_gs2_b01_fapar0
                        ]
    beta_strs = [
                 "0.00001",
                 "0.001",
                 "0.002",
                 "0.01"
                 ]
    make_comparison_plots_many(stella_sim_longnames,
                               gs2_sim_longnames,
                               beta_strs, "fapar=0/", plot_apar=False, plot_bpar=True)
    return


def plot_fbpar0():
    """ """
    stella_sim_longnames = [
                            sim_st_b00001_fbpar0,
                            sim_st_b001_fbpar0,
                            sim_st_b002_fbpar0,
                            sim_st_b003_fbpar0,
                            sim_st_b004_fbpar0,
                            sim_st_b005_fbpar0,
                            sim_st_b01_fbpar0
                            ]

    gs2_sim_longnames = [
                            sim_gs2_b00001_fbpar0,
                            sim_gs2_b001_fbpar0,
                            sim_gs2_b002_fbpar0,
                            sim_gs2_b003_fbpar0,
                            sim_gs2_b004_fbpar0,
                            sim_gs2_b005_fbpar0,
                            sim_gs2_b01_fbpar0
                        ]
    beta_strs = [
                 "0.00001",
                 "0.001",
                 "0.002",
                 "0.003",
                 "0.004",
                 "0.005",
                 "0.01"
                 ]
    make_comparison_plots_many(stella_sim_longnames,
                               gs2_sim_longnames,
                               beta_strs, "fbpar=0/", plot_apar=True, plot_bpar=False)

def make_all_plots():
    """ """
    plot_fbpar0()
    plot_fapar_fbpar_on()
    plot_fapar0()
    return

if __name__ == "__main__":
    ## Compare

    # analyse_fbpar0_results()
    # plot_geometry()
    # make_low_beta_fbpar0_plots()
    # analyse_fapar0_results()
    # plot_gvmus_for_fbpar0()
    # analyse_fapar0_changing_vpares()
    # make_all_plots()
    #plot_gzvs_for_fbpar0()
    analyse_fbpar0_beta0001_results()
