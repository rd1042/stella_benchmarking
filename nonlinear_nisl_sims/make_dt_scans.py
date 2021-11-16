""" """

import sys
sys.path.append("../postprocessing_tools")
sys.path.append("../generate_sims")
from plotting_helper import make_comparison_plots, plot_gmvus, plot_gzvs
from plotting_helper import make_comparison_plots_leapfrog_poster
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
from extract_sim_data import get_omega_data
from make_param_scans import make_input_files_1d
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

def make_dt_scan(folder_name, dt_vals):
    """ """

    template_name = folder_name + "/template_input_file"
    make_input_files_1d(template_name, None, 'delt', dt_vals, 'delt', folder=folder_name)
    shutil.copy('../templates_and_scripts/run_stella_local.sh', folder_name); os.chmod((folder_name + '/run_stella_local.sh'), 0o777)
    print("folder contents: " + folder_name)
    os.system('ls -ltr ' + folder_name)
    return

if __name__ == "__main__":

    dt_vals1 = np.logspace(-4, 0, 10)
    dt_vals2 = np.linspace(1.1, 2, 9)
    dt_vals3 = np.logspace(0.5, 4, 30)
    dt_vals = np.concatenate((dt_vals1, dt_vals2, dt_vals3))
    folder_name_rk3 = "sims_rk3_vexb_x1_dt_scan"
    folder_name_nisl_exact = "sims_nisl_vexb_x1_exact_start_dt_scan"
    make_dt_scan(folder_name_nisl_exact, dt_vals)
    print("Done")
