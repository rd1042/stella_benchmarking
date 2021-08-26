""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_beta_scan_plots, make_comparison_plots, plot_gmvus, plot_gzvs
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
from extract_sim_data import find_bpar_phi_ratio, find_apar_phi_ratio
import matplotlib.pyplot as plt
import numpy as np
import re

sim_st_b001_fbpar0_no_drive = "input_zero_drive"
sim_debugging = "input_debugging"
sim_debugging_default_res = "input_debugging_default_res"

def look_at_rhs_from_textfiles():
    """ """

    rhs1760_name = "ivmu=1760_zero_drive.txt"
    rhs1768_name = "ivmu=1768_zero_drive.txt"
    gin1760_name = "gin_ivmu=1760_zero_drive.txt"
    gin1768_name = "gin_ivmu=1768_zero_drive.txt"

    fig = plt.figure()
    ax1 = fig.add_subplot()

    for filename in [rhs1760_name,   rhs1768_name,     gin1760_name, gin1768_name]:
        values = []
        file1 = open(filename, "r")
        filetext = file1.read()
        file1.close()
        rhs_textblock = re.split("=", filetext.strip())[1]
        entries = re.split("\s+", rhs_textblock.strip())
        for entry in entries:
            # Get rid of first and last bracket
            entry = entry[1:-1]
            [real_str, im_str] = re.split(",", entry)
            val = complex(float(real_str), float(im_str))
            values.append(val)

            value_array = np.array(values)
            ax1.plot(np.arange(0, len(value_array)), value_array.real)
            ax1.plot(np.arange(0, len(value_array)), value_array.imag)

            plt.show()
    return

def plot_phi_t(outnc_longname, save_folder, step=1):
    """ """
    [t, z, phi_vs_t] = extract_data_from_ncdf(outnc_longname, "t", "zed", "phi_vs_t")

    print("len(t) = ", len(t))
    print("len(z) = ", len(z))
    #print("phi_vs_t.shape = ", phi_vs_t.shape)  # (t, tube, z, ky, kx) I think

    for t_idx in range(0, len(t), step):
    #for t_idx in [len(t)-1]:
        fig = plt.figure()
        ax1 = fig.add_subplot()
        norm_fac = np.max(abs(phi_vs_t[t_idx, 0, :, 0, 0]))
        ax1.plot(z, phi_vs_t[t_idx, 0, :, 0, 0].real/norm_fac)
        ax1.plot(z, phi_vs_t[t_idx, 0, :, 0, 0].imag/norm_fac)
        ax1.plot(z, abs(phi_vs_t[t_idx, 0, :, 0, 0])/norm_fac)
        ax1.grid(True)
        ax1.set_ylim(-1.1,1.1)
        #ax1.set_xlim(0, 10)
        plt.savefig(save_folder + f'/{t_idx:03}.png')
        plt.close()

    return


def plot_for_marconi_sims():
    """ """
    low_res_folder = "low_res_marconi"
    high_res_folder = "high_res_marconi"
    low_res_fapar0_folder = "low_res_marconi_fapar0"
    high_res_fapar0_folder = "high_res_marconi_fapar0"
    low_res_outnc = low_res_folder + "/input_low_res_marconi.out.nc"
    high_res_outnc = high_res_folder + "/input_high_res_marconi.out.nc"
    low_res_outnc_fapar0 = low_res_fapar0_folder + "/input_low_res_marconi_fapar0.out.nc"
    high_res_outnc_fapar0 = high_res_fapar0_folder + "/input_high_res_marconi_fapar0.out.nc"

    plot_phi_t(low_res_outnc, low_res_folder, step=10)
    plot_phi_t(high_res_outnc, high_res_folder, step=10)
    plot_phi_t(low_res_outnc_fapar0, low_res_fapar0_folder, step=10)
    plot_phi_t(high_res_outnc_fapar0, high_res_fapar0_folder, step=10)

    return

if __name__ == "__main__":

    print("Hello world")
    sim_st_b001_fbpar0_no_drive_outnc = sim_st_b001_fbpar0_no_drive + ".out.nc"
    print("sim_st_b001_fbpar0_no_drive_outnc = ", sim_st_b001_fbpar0_no_drive_outnc)
    # sim_debugging_outnc = sim_debugging + ".out.nc"
    # sim_debugging_default_res_outnc = sim_debugging_default_res + ".out.nc"
    # sim_debugging_folder = "images_input_debugging"
    # debug_save_folder_fapar0 = "images_debugging_fapar0"
    # debug_default_res_folder = "images_debugging_default_res"
    # plot_phi_t(sim_debugging_outnc, sim_debugging_folder, step=10)
    # plot_phi_t(sim_debugging_default_res_outnc, debug_default_res_folder, step=10)
    plot_for_marconi_sims()
