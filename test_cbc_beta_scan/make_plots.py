""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_beta_scan_plots, make_comparison_plots
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import matplotlib.pyplot as plt
import numpy as np

def analyse_fbpar0_results():
    """Compare omega(t) and phi(z), apar(z)
    between GS2 and stella results """

    print("Hello world")
    beta_strs = [
                 "0.01",
                 #"0.02",
                 #"0.03",
                 #"0.04",
                 "0.05",
                 "0.10",
                 ]
    stella_sim_longnames = [
                            "stella_beta0.001_fbpar0/input",
                            #"stella_beta0.002_fbpar0/input",
                            #"stella_beta0.003_fbpar0/input",
                            #"stella_beta0.004_fbpar0/input",
                            "stella_beta0.005_fbpar0/input",
                            "stella_beta0.010_fbpar0/input",
                            ]
    gs2_sim_longnames = [
                            "gs2_beta_scan_fbpar0/_0.0010",
                            #"gs2_beta_scan_fbpar0/_0.0020",
                            #"gs2_beta_scan_fbpar0/_0.0030",
                            #"gs2_beta_scan_fbpar0/_0.0040",
                            "gs2_beta_scan_fbpar0/_0.0050",
                            "gs2_beta_scan_fbpar0/_0.0100",
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
                 "0.010",
                 ]
    stella_sim_longnames = [
                            "stella_beta0.00001_fapar0/input",
                            "stella_beta0.001_fapar0/input",
                            "stella_beta0.002_fapar0/input",
                            "stella_beta0.010_fapar0/input",
                            ]
    gs2_sim_longnames = [
                            "gs2_beta_scan_fapar0/_0.00001",
                            "gs2_beta_scan_fapar0/_0.0010",
                            "gs2_beta_scan_fapar0/_0.0020",
                            "gs2_beta_scan_fapar0/_0.0100",
                            ]
    # for beta_idx in range(0, len(beta_strs)):
    #     stella_sim_longname = stella_sim_longnames[beta_idx]
    #     gs2_sim_longname = gs2_sim_longnames[beta_idx]
    #     beta_str = beta_strs[beta_idx]
    #     make_comparison_plots([
    #                            stella_sim_longname,
    #                            gs2_sim_longname,
    #                            ],
    #                           [
    #                            "stella",
    #                            "GS2",
    #                            ],
    #                           "./beta_" + beta_str + "_fapar0",
    #                           sim_types=[
    #                                      "stella",
    #                                      "gs2",
    #                                      ],
    #                            plot_bpar=True,
    #                            )

    plot_gmvus("stella_beta0.001_fapar0/input.out.nc")
    return


def plot_gmvus(stella_outnc_longname):
    """ """
    t, z, mu, vpa, gds2, gds21, gds22, bmag, gradpar, gvmus = extract_data_from_ncdf(stella_outnc_longname,
                                    "t", 'zed', "mu", "vpa", 'gds2', 'gds21', 'gds22', 'bmag', 'gradpar', 'gvmus')
    print("len(t)", "len(z), len(mu), len(vpa) = ", len(t), len(z), len(mu), len(vpa))
    #print("gvmus.shape = ", gvmus.shape)

    sys.exit()

    ## Code to plot g for stella
    gvmus = gvmus[-1]   # spec, mu, vpa
    fig = plt.figure(figsize=[10,10])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    counter=0

    for mu_idx in range(0, len(mu)):
        counter += 1
        g_ion_vpa = gvmus[0, mu_idx, :]
        g_electron_vpa = gvmus[1, mu_idx, :]
        ax1.plot(vpa, g_ion_vpa, label="mu={:.3f}".format(mu[mu_idx]))
        ax2.plot(vpa, g_electron_vpa, label="mu={:.3f}".format(mu[mu_idx]))

        if counter == 5:
            for ax in [ax1, ax2]:
                ax.grid(True)
                ax.legend(loc="best")
            ax2.set_xlabel("vpa")
            ax1.set_ylabel(r"$g_{ion}$")
            ax2.set_ylabel(r"$g_{electron}$")
            plt.show()
            fig = plt.figure(figsize=[10,10])
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            counter=0

    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.legend(loc="best")
    ax2.set_xlabel("vpa")
    ax1.set_ylabel(r"$g_{ion}$")
    ax2.set_ylabel(r"$g_{electron}$")
    plt.show()

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


def make_low_beta_plots():
    """ """
    gs2_sim = "gs2_beta_scan_fbpar0/_0.0000"
    #gs2_sim = "gs2_beta_scan_fbpar0/_0.0010"
    sim1 = "stella_beta0_fbpar0/input"
    sim2 = "stella_beta0.00001_fbpar0/input"

    make_comparison_plots([
                           gs2_sim,
                           sim1,
                           sim2,
                           ],
                          [
                           "gs2 beta=0",
                           #"gs2 beta=0.001",
                           "beta=0",
                           "beta=0.0001",
                           ],
                          "./low_beta",
                          sim_types=[
                                     "gs2",
                                     "stella",
                                     "stella",
                                     ],
                           plot_apar=False,
                           )
    plot_gmvus(sim2 + ".out.nc")

    return


if __name__ == "__main__":
    ## Compare
    stella_sim_longnames = [
                            "stella_beta0.000/input",
                            "stella_beta0.005/input",
                            "stella_beta0.010/input",
                            "stella_beta0.015/input",
                            "stella_beta0.020/input",
                            "stella_beta0.025/input",
                            "stella_beta0.030/input",
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
    gs2_pickle = "gs2_beta_scan/omega_values.pickle"

    # make_beta_scan_plots(stella_sim_longnames,
    #                      stella_beta_vals,
    #                       "./test_cbc_beta_scan",
    #                       gs2_pickle=gs2_pickle
    #                      )

    stella_sim_longnames = [
                            "stella_beta0.000/input",
                            "stella_beta0.001_fbpar0/input",
                            "stella_beta0.002_fbpar0/input",
                            "stella_beta0.003_fbpar0/input",
                            "stella_beta0.004_fbpar0/input",
                            "stella_beta0.005_fbpar0/input",
                            ]
    stella_beta_vals = [
                        0.0,
                        0.001,
                        0.002,
                        0.003,
                        0.004,
                        0.005,
                        ]

    gs2_pickle = "gs2_beta_scan_fbpar0/omega_values.pickle"

    # make_beta_scan_plots(stella_sim_longnames,
    #                      stella_beta_vals,
    #                       "./test_cbc_beta_scan_fbpar0",
    #                      gs2_pickle=gs2_pickle
    #                      )
    # analyse_fbpar0_results()
    # plot_geometry()
    # make_low_beta_plots()
    analyse_fapar0_results()
