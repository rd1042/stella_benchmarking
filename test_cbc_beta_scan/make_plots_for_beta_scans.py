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



def compare_beta_scans():
    """ """
    # Find all the input files in the folder.
    # NB might be more failsafe to find all sims with a .in, .out and
    # .omega file, but this is more work.
    sim_infile_longnames = glob.glob(stella_fapar1_fbpar0_me1_folder + "/*.in")
    unsorted_stella_longnames = []
    beta_vals = []
    for sim_idx, sim_infile_longname in enumerate(sim_infile_longnames):
        # Get the sim longanme and beta.
        stella_longname = re.split(".in", sim_infile_longname)[0]
        unsorted_stella_longnames.append(stella_longname)
        beta_str = re.split("beta_", stella_longname)[-1]
        beta_vals.append(float(beta_str))

    # Sort into ascending order of beta
    beta_vals = np.array(beta_vals)
    sort_idxs = np.argsort(beta_vals)
    beta_vals = beta_vals[sort_idxs]
    stella_longnames = []
    for sort_idx in sort_idxs:
        stella_longnames.append(unsorted_stella_longnames[sort_idx])
    print("stella_longnames = ", stella_longnames)
    save_name = "fapar1_fbpar0"
    make_beta_scan_plots(stella_longnames, beta_vals, save_name,
            gs2_pickle=None,  plot_apar=True, plot_bpar=False, plot_format=".png")



if __name__ == "__main__":
    compare_beta_scans()
