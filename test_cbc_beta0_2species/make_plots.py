""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots



if __name__ == "__main__":
    ## Compare
    master_sim_longname = "master_cmiller_es_2species_ypi/cmiller_electrostatic_2species"
    em_1field_sim_longname = "electromagnetic_1field/cmiller_beta0_2species_explicit_emsields_0"
    em_3field_sim_longname = "electromagnetic_3fields/cmiller_beta0_2species_explicit"
    make_comparison_plots([master_sim_longname,
                           em_1field_sim_longname,
                           em_3field_sim_longname,
                           ],
                          ["master",
                           "feature/electromagnetic, 1 field",
                           "feature/electromagnetic, 3 fields",
                           ],
                          "./test_cbc_beta0_2species")
