""" """

from nonlinear_stella_sim_visualiser import make_phi2_kxky_modes_pics_single_mode
from nonlinear_stella_sim_visualiser import make_phi_kxky_modes_pics
from nonlinear_stella_sim_visualiser import make_phi2_kxky_modes_pics_for_each_z

if __name__ == "__main__":
    #make_phi2_kxky_modes_pics_single_mode("sims/nisl/example_nisl_nonlinear_only_vexb_x16_vexb_y0_first_step_exact_nwrite1.out.nc")
    # make_phi_kxky_modes_pics("sims/nisl/example_nisl_nonlinear_only_vexb_x16_single_mode_first_step_exact_nwrite1.out.nc")
    #make_phi2_kxky_modes_pics_single_mode("sims/nisl/example_nisl_nonlinear_only_vexb_x10_single_mode_first_step_exact_nwrite1.out.nc")
    #make_phi2_kxky_modes_pics_single_mode("sims/nisl/example_nisl_nonlinear_only_vexb_x16_single_mode_nwrite1.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("sims/nisl/example_nisl_nonlinear_only_vexb10_for_visualisation.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("sims/nisl/example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite51.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("sims/nisl/example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite1.out.nc")
    # make_phi2_kxky_modes_pics_for_each_z("sims/rk3/example_nisl_nonlinear_only_vexb10_single_mode.out.nc")
    # make_phi2_kxky_modes_pics_for_each_z("sims/nisl/example_nisl_nonlinear_only_vexb10_single_mode.out.nc")
    # make_phi2_kxky_modes_pics_for_each_z("sims/nisl/example_nisl_nonlinear_only_vexb10_single_mode_kxky0.667.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("sims/nisl/example_nisl_nonlinear_only_vexb1_for_visualisation.out.nc")

    #make_phi_kxky_modes_pics("sims/nisl/example_nisl_nonlinear_only_vexb1_for_visualisation.out.nc")
    # make_phi2_kxky_modes_pics_single_mode("sims/nisl/example_nisl_nonlinear_only_vexb10_for_visualisation.out.nc")
    # make_phi2_kxky_modes_pics_single_mode("sims/nisl/example_nisl_nonlinear_only_vexb10_for_visualisation_new.out.nc")
    #make_phi_kxky_modes_pics("sims/nisl/example_nisl_nonlinear_only_vexb_x100_single_mode_nwrite1.out.nc")

    # A look at instancing only zonal modes
    #make_phi_kxky_modes_pics("sims/rk3/example_rk3_nonlinear_only.out.nc")
    #make_phi_kxky_modes_pics("sims/rk3/example_rk3_nonlinear_only_zonal_only.out.nc")
    make_phi_kxky_modes_pics("sims/nisl/example_nisl_nonlinear_only_zonal_only.out.nc")

    ### Plots for the sims with actual ExB velocities
    #make_phi_kxky_modes_pics("sims/rk3/example_nisl_nonlinear_only_dt5e-2.out.nc")
    #make_phi_kxky_modes_pics("sims/leapfrog/example_leapfrog_nonlinear_only_dt5e-2.out.nc", log=True)
    #make_phi_kxky_modes_pics("sims/nisl/example_nisl_nonlinear_only_dt5e-2.out.nc")
    #make_phi_kxky_modes_pics("sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000_2species.out.nc")
    #make_phi_kxky_modes_pics("sims/leapfrog/example_leapfrog_nonlinear_only_dt0.01.out.nc")
