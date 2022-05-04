"""Making plots to compare key quantities from different simulations
(i.e. different choice of how to treat the nonlinear term)
Includes sims where we have only the nonlinear term, and sims where we have the
full GKE being advanced.
"""

from nonlinear_stella_sim_visualiser import make_phi2_kxky_modes_pics_single_mode_multiple_sims
from nonlinear_stella_sim_visualiser import make_phi_kxky_modes_pics_multiple_sims
from nonlinear_stella_sim_visualiser import make_phi_kxky_modes_pics_multiple_sims_all_z
from nonlinear_stella_sim_visualiser import make_phi2_kxky_modes_pics_multiple_sims
from nonlinear_stella_sim_visualiser import make_phi2_ky_modes_pics_multiple_sims
from nonlinear_stella_sim_visualiser import make_phi_ky_modes_pics_multiple_sims
from nonlinear_stella_sim_visualiser import make_phi2_ky_modes_pics_multiple_sims_for_talk
from nonlinear_stella_sim_visualiser import plot_phi2t_for_rk3_folder, plot_phi2t_for_folder

if __name__ == "__main__":

    #make_phi2_kxky_modes_pics("example_rk3_nonlinear_only_vexb1_for_visualisation.out.nc")
    #make_phi2_kxky_modes_pics("example_rk3_nonlinear_only_vexb10_for_visualisation_longer_time.out.nc")


    ### Comparing RK3, Leapfrog, NISL with same timestep=0.01. Adiabatic electrons,
    ### nonlinear only
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep50000.out.nc",
    #                                 "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.1_nstep10000.out.nc",
    #                                 ],
    #                                 ["RK3, delt=1E-2",
    #                                 "Leapfrog, delt=1E-2",
    #                                 "NISL, delt=1E-2",
    #                                 "NISL, delt=1E-1"],
    #                                 ["-", "-.", "--", "--"])

    ### make_phi2_ky_modes_pics_multiple_sims Doesn't work - we've got phi(kx,ky,t), but not phi2
    # make_phi_ky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.1_nstep10000.out.nc",
    #                                 ],
    #                                 ["RK3, delt=1E-2",
    #                                 "Leapfrog, delt=1E-2",
    #                                 "NISL, delt=1E-2",
    #                                 "NISL, delt=1E-1"],
    #                                 ["black", "red", "blue", "green"],
    #                                 min_phi=1E-5, transparency_val=0.3)
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 #"sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000_add_in_fourier_space.out.nc",
    #                                 ],
    #                                 ["RK3, delt=1E-2",
    #                                 "Leapfrog, delt=1E-2",
    #                                 "NISL, delt=1E-2",
    #                                 #"NISL, F, delt=1E-2",
    #                                 ],
    #                                 ["-", "-.", "--", ":"])
    ############################################################################
    ### Comparing RK3, Leapfrog, NISL with same timestep=0.01. Adiabatic electrons,
    ### nonlinear only
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_zonal_only.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_zonal_only.out.nc"],
    #                                 ["RK3, zonal only",
    #                                 "NISL, zonal only"],
    #                                 ["-", "--"])
    # make_phi_kxky_modes_pics_multiple_sims_all_z(["sims/rk3/example_rk3_nonlinear_only_single_mode.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_single_mode.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_single_mode_add_in_fourier_space.out.nc"
    #                                 ],
    #                                 ["RK3, single mode",
    #                                 "NISL, single mode",
    #                                 "NISL, single mode, add source in fourier space"],
    #                                 ["black", "red", "blue"])
    # make_phi_kxky_modes_pics_multiple_sims_all_z(["sims/rk3/example_rk3_nonlinear_only_single_mode_backup.out.nc",
    #                                 "sims/rk3/example_rk3_nonlinear_only_single_mode.out.nc",
    #                                 ],
    #                                 ["RK3, single mode (old)",
    #                                 "RK3, single mode (new)"
    #                                 ],
    #                                 ["black", "red"
    #                                 ])
    ############################################################################
    ############################################################################
    ### Comparing RK3, Leapfrog, NISL with same timestep=0.01. Kinetic electrons,
    ### nonlinear only
    ############################################################################
    make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_2species.out.nc",
                                    "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep2000_2species.out.nc",
                                    "sims/nisl/example_nisl_nonlinear_only_dt0.01_2species.out.nc"],
                                    ["RK3, delt=1E-2",
                                    "Leapfrog, delt=1E-2",
                                    "NISL, delt=1E-2"],
                                    ["-", "-.", "--"])
    # Same, but shorter simulation time
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep2000_2species.out.nc",
    #                                 "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep2000_2species.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep2000_2species.out.nc"],
    #                                 ["RK3, delt=1E-2",
    #                                 "Leapfrog, delt=1E-2",
    #                                 "NISL, delt=1E-2"],
    #                                 ["-", "-.", "--"])
    # Same, but longer NISL simulation time
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_2species.out.nc",
    #                                 "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep2000_2species.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep100000_2species.out.nc"],
    #                                 ["RK3, delt=1E-2",
    #                                 "Leapfrog, delt=1E-2",
    #                                 "NISL, delt=1E-2"],
    #                                 ["-", "-.", "--"])
    ############################################################################

    ############################################################################
    ### Comparing RK3, Leapfrog, NISL with same timestep=0.01, different amounts of
    ### padding. Adiabatic electrons, nonlinear only,
    ############################################################################
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/extra_padding_sims/example_rk3_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/extra_padding_sims/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc"],
    #                                 ["RK3, standard",
    #                                 "RK3, extra padding",
    #                                 "NISL, standard",
    #                                 "NISL, extra padding"],
    #                                 ["-", "-.", "--", ":"])
    ### The same, but kinetic electrons
    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep10000_2species.out.nc",
    #                                 "sims/extra_padding_sims/example_rk3_nonlinear_only_dt0.01_nstep10000_2species.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_2species.out.nc",
    #                                 "sims/extra_padding_sims/example_nisl_nonlinear_only_dt0.01_2species.out.nc"],
    #                                 ["RK3, standard",
    #                                 "RK3, extra padding",
    #                                 "NISL, standard",
    #                                 "NISL, extra padding"],
    #                                 ["-", "-.", "--", "-"])
    ############################################################################

    ############################################################################
    ### Comparing forcing the kx=ky=0 mode to zero, vs not.
    ############################################################################
    # make_phi_kxky_modes_pics_multiple_sims([
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000_not_zeroed.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 ],
    #                                 [
    #                                 "NISL, standard",
    #                                 "NISL, g(kx=ky=0)=0"],
    #                                 ["-", "-.", "--", "-"])
    ############################################################################

    # make_phi_kxky_modes_pics_multiple_sims(["sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep10000.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc"],
    #                                 ["RK3, delt=1E-2",
    #                                 "Leapfrog, delt=1E-2",
    #                                 "NISL, delt=1E-2"],
    #                                 ["-", "-.", "--"])
    # make_phi_kxky_modes_pics_multiple_sims(["sims/nisl/example_nisl_nonlinear_only_dt5e-2.out.nc",
    #                                 "sims/nisl/example_nisl_nonlinear_only_dt0.01_longer_time.out.nc",
    #                                 #"sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc"
    #                                 ],
    #                                 ["NISL, delt=5E-2",
    #                                 "NISL, delt=1E-2",
    #                                 #"NISL, delt=1E-2"
    #                                 ],
    #                                 ["-", "-.", "--"])


    #compare_sims_with_different_nwrite()
    #examine_initialisation()


    # plot_phi2t_for_rk3_folder("sims_rk3_vexb_x1_dt_scan", 10)
    # plot_phi2t_for_folder("sims_nisl_vexb_x1_exact_start_dt_scan", 10)

    ############################################################################
    ### Full simulaiton
    ############################################################################
    # make_phi2_kxky_modes_pics_multiple_sims(["../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_master/input.out.nc",
    #                                          "../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl/input.out.nc"
    #                                          ],
    #                                          ["RK3",
    #                                          "NISL"],
    #                                          ["-",
    #                                          "-."])

    ### Comparing full sims for RK3 vs NISL
    # Adiabatic electrons
    # make_phi2_ky_modes_pics_multiple_sims(["../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_master/input.out.nc",
    #                                          "../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl/input.out.nc"
    #                                          ],
    #                                          ["RK3",
    #                                          "NISL"],
    #                                          ["black", "red"])
    # Kinetic electrons
    # make_phi2_ky_modes_pics_multiple_sims_for_talk(["../test_cbc_nonlinear_beta0/stella_nonlinear_2species_master/input.out.nc",
    #                                          "../test_cbc_nonlinear_beta0/stella_nonlinear_2species_nisl/input.out.nc"
    #                                          ],
    #                                          ["explicit (RK3) scheme",
    #                                          "Non-interpolating SL"],
    #                                          ["black", "red"])
