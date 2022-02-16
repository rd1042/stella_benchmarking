""" """

from nonlinear_stella_sim_visualiser import make_phi2_t_movie
from nonlinear_stella_sim_visualiser import make_phi_t_movie
from nonlinear_stella_sim_visualiser import make_phi_z_movie
from nonlinear_stella_sim_visualiser import make_phi_xyz_plots
from nonlinear_stella_sim_visualiser import make_phi_z_t_movie


if __name__ == "__main__":

    ## Images of nonlinear sims
    #make_phi_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl/input_write_phi.out.nc",extra_upsample_fac=1)
    make_phi_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_master/input.out.nc",extra_upsample_fac=2)
    #make_phi_xyz_plots("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_master/input.out.nc")
    #make_phi_z_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_master/input.out.nc", t_idxs=[0,4,10,12,14,16,18,20,24])
    #make_phi_z_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl/input_write_phi.out.nc", t_idxs=[50,70,80,90,100])
    #make_phi2_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_2species_nisl/input.out.nc")
    #make_phi_z_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl_from_restart/input.out.nc", t_idxs=[0,1,2])
    # make_phi2_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl/input.out.nc")
    #make_phi_t_movie("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_nisl/input.out.nc", extra_upsample_fac=2)
    #make_phi_t_movie("sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc")
    #make_phi_t_movie("sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01_nstep10000.out.nc")


    #make_phi2_t_movie("example_rk3_nonlinear_only_vexb1_for_visualisation.out.nc")
    #make_phi2_t_movie("example_rk3_nonlinear_only_vexb10_for_visualisation_longer_time.out.nc")
    #make_phi2_t_movie("example_rk3_nonlinear_only_vexb_x0_y10_single_mode.out.nc")
