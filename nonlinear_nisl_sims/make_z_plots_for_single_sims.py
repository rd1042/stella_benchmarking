""" """

from nonlinear_stella_sim_visualiser import plot_phiz_for_zonal_mode


if __name__ == "__main__":
    #plot_phiz_for_zonal_mode("sims/rk3/example_rk3_nonlinear_only_dt0.01_nstep10000.out.nc")
    plot_phiz_for_zonal_mode("../test_cbc_nonlinear_beta0/stella_nonlinear_adiabatic_master/input.out.nc")
    #plot_phiz_for_zonal_mode("sims/nisl/example_nisl_nonlinear_only_dt0.01_nstep10000.out.nc")
