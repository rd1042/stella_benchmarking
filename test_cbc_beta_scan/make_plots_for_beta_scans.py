""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots, plot_gmvus, plot_gzvs, make_beta_scan_plots
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

## Define folder names here
# stella
stella_fapar0_fbpar1_me1_folder = "stella_fapar0_fbpar1_me1_beta_scan"
stella_fapar1_fbpar0_me1_folder = "stella_fapar1_fbpar0_me1_beta_scan"
stella_fapar1_fbpar1_me1_folder = "stella_fapar1_fbpar1_me1_beta_scan"


# gs2
gs2_fapar0_fbpar1_me1_folder = "gs2_fapar0_fbpar1_me1_beta_scan"
gs2_fapar1_fbpar0_me1_folder = "gs2_fapar1_fbpar0_me1_beta_scan"
gs2_fapar1_fbpar1_me1_folder = "gs2_fapar1_fbpar1_me1_beta_scan"



def get_sim_longnames(folder_longname):
    """ """
    # Find all the input files in the folder.
    # NB might be more failsafe to find all sims with a .in, .out and
    # .omega file, but this is more work.
    sim_infile_longnames = glob.glob(folder_longname + "/*.in")
    unsorted_longnames = []
    beta_vals = []
    for sim_idx, sim_infile_longname in enumerate(sim_infile_longnames):
        # Get the sim longanme and beta.
        sim_longname = re.split(".in", sim_infile_longname)[0]
        unsorted_longnames.append(sim_longname)
        beta_str = re.split("beta_", sim_longname)[-1]
        beta_vals.append(float(beta_str))

    # Sort into ascending order of beta
    beta_vals = np.array(beta_vals)
    sort_idxs = np.argsort(beta_vals)
    beta_vals = beta_vals[sort_idxs]
    sim_longnames = []
    for sort_idx in sort_idxs:
        sim_longnames.append(unsorted_longnames[sort_idx])

    return beta_vals, sim_longnames

def compare_beta_scans(stella_folder, gs2_folder, save_name, plot_apar=False, plot_bpar=False):
    """ """
    stella_beta, stella_longnames = get_sim_longnames(stella_folder)
    gs2_beta, gs2_longnames = get_sim_longnames(gs2_folder)

    print("stella_longnames = ", stella_longnames)
    print("gs2_longnames = ", gs2_longnames)

    # Expect gs2_beta = stella_beta; stop if not (to implement: do something
    # clever in this case)
    if np.max(abs(stella_beta - gs2_beta)) > 1e-4:
        print("Error! GS2 beta != stella beta . Stopping")
        print("stella_beta = ", stella_beta)
        print("gs2_beta = ", gs2_beta)
        sys.exit()
    make_beta_scan_plots(stella_longnames, gs2_longnames, gs2_beta, save_name,
            gs2_pickle=None,  plot_apar=True, plot_bpar=False, plot_format=".png")

def plot_different_beta_scans():
    """ """

    compare_beta_scans(stella_fapar1_fbpar0_me1_folder,
                       gs2_fapar1_fbpar0_me1_folder,
                       "images/fapar1_fbpar0")
    compare_beta_scans(stella_fapar1_fbpar1_me1_folder,
                       gs2_fapar1_fbpar1_me1_folder,
                       "images/fapar1_fbpar1")
    compare_beta_scans(stella_fapar0_fbpar1_me1_folder,
                       gs2_fapar0_fbpar1_me1_folder,
                       "images/fapar0_fbpar1")

    return

if __name__ == "__main__":
    plot_different_beta_scans()
