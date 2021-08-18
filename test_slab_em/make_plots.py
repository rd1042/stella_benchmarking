"""Make plots for electromagnetic slab sims"""

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_beta_scan_plots, make_comparison_plots, plot_gmvus, plot_gzvs
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
from extract_sim_data import find_bpar_phi_ratio, find_apar_phi_ratio
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import pickle
# import ast

IMAGE_DIR = "./images/"

basic_em_sim = "stella_sims/input_slab_ky0.1_explicit"
mandell_beta1_kperp1 = "stella_sims/input_slab_ky0.1_explicit_mandell1"
mandell_beta1_kperp1_long_t = "stella_sims/input_slab_ky0.1_explicit_mandell2"
mandell_beta1_kperp1_long_t_marconi = "mandell_sims/input_slab_ky0.1_beta1"

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

    t_chop = int(len(t)/2)
    print("len(t), t_chop, t[t_chop] = ", len(t), t_chop, t[t_chop] )
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(t[:t_chop], phi_vs_t[:t_chop,int(len(z)*0.5)].real, label=("z=" + str(z[int(len(z)*0.5)]) ))
    ax2.plot(t[:t_chop], phi_vs_t[:t_chop,int(len(z)*0.5)].imag)
    ax1.plot(t[:t_chop], phi_vs_t[:t_chop,int(len(z)*0.2)].real, label=("z=-1.9" ))
    ax2.plot(t[:t_chop], phi_vs_t[:t_chop,int(len(z)*0.2)].imag)
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


def find_ksaw_properties(phi_vs_t_file):
    """ """
    file = open(phi_vs_t_file, "rb")
    [z, t, phiz_final, phit_mid] = pickle.load(file)
    file.close()
    ### Lines are :
    #  "z"
    #  z
    #  "t"
    #  t
    #  "phi_vs_t[-1,:]"
    #  phi_vs_t[-1,:]
    #  "phi_vs_t[:,int(len(z)*0.5)]"
    #  phi_vs_t[:,int(len(z)*0.5)]

    # z = np.array(ast.literal_eval(lines[1]))
    # t = np.array(ast.literal_eval(lines[3]))
    # phiz_final = ast.literal_eval(lines[5])
    # print("phiz_final = ", phiz_final)
    # sys.exit()
    # phit_mid = np.array(ast.literal_eval(lines[7]))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(z, phiz_final.real)
    ax2.plot(z, phiz_final.imag)
    ax2.set_xlabel("z")
    ax2.set_ylabel("Im(phi)")
    ax1.set_ylabel("Re(phi)")
    fig.suptitle(phi_vs_t_file)
    for ax in [ax1, ax2]:
        ax.grid(True)
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(t, phit_mid.real)
    ax2.plot(t, phit_mid.imag)
    ax2.set_xlabel("t")
    ax2.set_ylabel("Im(phi)")
    ax1.set_ylabel("Re(phi)")
    ax1.legend(loc="best")
    fig.suptitle(phi_vs_t_file)
    for ax in [ax1, ax2]:
        ax.grid(True)
    plt.show()


    return

def benchmark_stella_vs_mandell():
    """ """
    phi_vs_t_longnames = glob.glob("mandell_sims/*.pickle")

    for file_longname in phi_vs_t_longnames:
        print("file_longname = ", file_longname)
        find_ksaw_properties(file_longname)
    return

def examine_second_sim():
    """ """
    examine_sim_output(mandell_beta1_kperp1_long_t_marconi)
    return

if __name__ == "__main__":
    print("Hello world")
    # examine_first_sim()
    # examine_second_sim()
    benchmark_stella_vs_mandell()
