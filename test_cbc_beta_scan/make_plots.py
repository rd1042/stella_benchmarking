""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_beta_scan_plots



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

    make_beta_scan_plots(stella_sim_longnames,
                         stella_beta_vals,
                          "./test_cbc_beta_scan",
                          gs2_pickle=gs2_pickle
                         )
