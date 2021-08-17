"""Make plots for electromagnetic slab sims"""

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_beta_scan_plots, make_comparison_plots, plot_gmvus, plot_gzvs
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
from extract_sim_data import find_bpar_phi_ratio, find_apar_phi_ratio
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DIR = "./images/"

basic_em_sim = "stella_sims/input_slab_ky0.1_explicit"
mandell_beta1_kperp1 = "stella_sims/input_slab_ky0.1_explicit_mandell1"
mandell_beta1_kperp1_long_t = "stella_sims/input_slab_ky0.1_explicit_mandell2"

def examine_sim_output(sim_longname):
    """ """
    outnc_longname = sim_longname + ".out.nc"
    ### Plot geometric terms
    ## Code to compare geometry between stella and gs2

    # view_ncdf_variables(outnc_longname)
    # ['code_info', 'nproc', 'nmesh', 'ntubes', 'nkx', 'nky', 'nzed_tot',
    # 'nspecies', 'nmu', 'nvpa_tot', 't', 'charge', 'mass', 'dens', 'temp',
    # 'tprim', 'fprim', 'vnew', 'type_of_species', 'theta0', 'kx', 'ky', 'mu',
    # 'vpa', 'zed', 'bmag', 'gradpar', 'gbdrift', 'gbdrift0', 'cvdrift',
    # 'cvdrift0', 'kperp2', 'gds2', 'gds21', 'gds22', 'grho', 'jacob', 'q',
    # 'beta', 'shat', 'jtwist', 'drhodpsi', 'phi2', 'phi_vs_t', 'bpar2',
    # 'gvmus', 'gzvs', 'input_file']

    ###### Plot geometry
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    z, gds2, gds21, gds22, bmag, gradpar = extract_data_from_ncdf(outnc_longname,
                                    'zed', 'gds2', 'gds21', 'gds22', 'bmag', 'gradpar')
    ax1.plot(z, gds2)
    ax1.plot(z, gds21)
    ax1.scatter(z, gds22)
    ax2.plot(z, bmag)
    ax2.plot(z, gradpar)
    plt.show()

    t, z, phi_vs_t = extract_data_from_ncdf(outnc_longname, "t", 'zed', 'phi_vs_t')
    # print("len(t) = ", len(t))
    # print("len(z) = ", len(z))
    # print("phi_vs_t.shape = ", phi_vs_t.shape)  # [n_time, 1 , n_z, 1, 1]
    phi_vs_t = phi_vs_t[:,0,:,0,0]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(z, phi_vs_t[-20,:].real, label="t[-20]")
    ax2.plot(z, phi_vs_t[-20,:].imag)
    ax1.plot(z, phi_vs_t[-10,:].real, label="t[-10]")
    ax2.plot(z, phi_vs_t[-10,:].imag)
    ax1.plot(z, phi_vs_t[-1,:].real, label="t[-1]")
    ax2.plot(z, phi_vs_t[-1,:].imag)
    ax2.set_xlabel("z")
    ax2.set_ylabel("Im(phi)")
    ax1.set_ylabel("Re(phi)")
    ax1.legend(loc="best")
    for ax in [ax1, ax2]:
        ax.grid(True)
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(t, phi_vs_t[:,int(len(z)*0.5)].real, label=("z=" + str(z[int(len(z)*0.5)]) ))
    ax2.plot(t, phi_vs_t[:,int(len(z)*0.5)].imag)
    ax1.plot(t, phi_vs_t[:,int(len(z)*0.2)].real, label=("z=-1.9" ))
    ax2.plot(t, phi_vs_t[:,int(len(z)*0.2)].imag)
    ax2.set_xlabel("t")
    ax2.set_ylabel("Im(phi)")
    ax1.set_ylabel("Re(phi)")
    ax1.legend(loc="best")
    for ax in [ax1, ax2]:
        ax.grid(True)
    plt.show()

    return

def examine_first_sim():
    """ """
    examine_sim_output(basic_em_sim)
    return

def benchmark_stella_vs_mandell():
    """ """
    examine_sim_output(mandell_beta1_kperp1_long_t)
    return


if __name__ == "__main__":
    print("Hello world")
    # examine_first_sim()
    benchmark_stella_vs_mandell()
