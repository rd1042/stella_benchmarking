""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots, plot_gmvus


def compare_stella_to_gs2():
    """ """
    master_sim_longname1 = "master_cmiller_es_2species_ypi/cmiller_electrostatic_2species"
    master_sim_longname2 = "master_cmiller_es_2species_marconi/increased_nvpa/input"
    master_sim_longname3 = "master_cmiller_es_2species_marconi/input"
    em_1field_sim_longname = "electromagnetic_1field/cmiller_beta0_2species_explicit_emsields_0"
    #em_3field_sim_longname = "electromagnetic_3fields/cmiller_beta0_2species_explicit"
    gs2_sim_longname = "gs2_electrostatic/cmiller_new_normal_0.0000"
    gs2new_sim_longname = "gs2_electrostatic_new/_0.0000"
    make_comparison_plots([
                    master_sim_longname1,
                    master_sim_longname2,
                    master_sim_longname3,
                    # em_1field_sim_longname,
                    # em_3field_sim_longname,
                    gs2_sim_longname,
                    gs2new_sim_longname,
                    ],
                    [
                    "stella master",
                    "stella master (new)",
                    "stella master (new new)",
                    #"feature/electromagnetic, 1 field",
                    #"feature/electromagnetic, 3 fields",
                    "GS2",
                    "GS2 (new)",
                    ],
                    "./test_cbc_beta0_2species",
                    sim_types=[
                    #"stella",
                    "stella",
                    "stella",
                    "stella",
                    #"stella",
                    "gs2",
                    "gs2",
                    ])
    return

def plot_g_for_stella_sim():
    """ """
    master_outnc_longname2 = "master_cmiller_es_2species_marconi/input.out.nc"
    plot_gmvus(master_outnc_longname2)
    return


if __name__ == "__main__":
    plot_g_for_stella_sim()
    # compare_stella_to_gs2()
