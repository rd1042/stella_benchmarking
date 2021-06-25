""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots, plot_gmvus, plot_gzvs
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import matplotlib.pyplot as plt
import numpy as np

stella_basecase_longname = "master_cmiller_es_2species_marconi/input"
stella_nperiod5_longname = "master_cmiller_es_2species_marconi/input5"
stella_nperiod5_ntheta128_longname = "master_cmiller_es_2species_marconi/input5_ntheta128"
stella_nperiod5_dt0001_longname = "master_cmiller_es_2species_marconi/input5_dt001"
stella_nperiod5_upwind02_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_0.2"
stella_nperiod5_upwind05_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_0.5"
stella_nperiod5_upwind1_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_1"
stella_nperiod5_upwind0_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_0"
stella_nperiod5_t_upwind0_longname = "master_cmiller_es_2species_marconi/input5_t_upwind0"
stella_nperiod5_t_upwind0_zvpa_upwind_0_longname = "master_cmiller_es_2species_marconi/input5_t_zvpa_upwind0"
stella_nperiod5_t_upwind0_zvpa_upwind_0_implicit_mirror_longname = "master_cmiller_es_2species_marconi/input5_t_zvpa_upwind0_implicit_mirror"
stella_nperiod5_longname = "master_cmiller_es_2species_marconi/input5"
stella_nperiod7_longname = "master_cmiller_es_2species_marconi/input7"
stella_nperiod9_longname = "master_cmiller_es_2species_marconi/input9"
stella_adiabatic_longname = "stella_adiabatic/input5"
stella_adiabatic_m001_longname = "stella_adiabatic/input5_mi0.01"
stella_adiabatic_m01_longname = "stella_adiabatic/input5_mi0.1"
stella_adiabatic_m05_longname = "stella_adiabatic/input5_mi0.5"
stella_adiabatic_m2_longname = "stella_adiabatic/input5_mi2"
stella_adiabatic_m10_longname = "stella_adiabatic/input5_mi10"

gs2_basecase_longname = "gs2_electrostatic_new/_0.0000"
gs2_dt001_longname = "gs2_electrostatic_new/beta0_dt0.01"
gs2_adiabatic_longname = "gs2_adiabatic/input"
gs2_adiabatic_m001_longname = "gs2_adiabatic/input_mi0.01"
gs2_adiabatic_m01_longname = "gs2_adiabatic/input_mi0.1"
gs2_adiabatic_m05_longname = "gs2_adiabatic/input_mi0.5"
gs2_adiabatic_m2_longname = "gs2_adiabatic/input_mi2"
gs2_adiabatic_m10_longname = "gs2_adiabatic/input_mi10"
gs2_eqarc_false_longname = "gs2_electrostatic_new/_0.0000_eqarc_false"
gs2_bakdif0_longname = "gs2_electrostatic_new/beta0_bakdif0"
gs2_bakdif0_fexpr05_longname = "gs2_electrostatic_new/beta0_bakdif0_fexpr0.5"
gs2_bakdif1_longname = "gs2_electrostatic_new/beta0_bakdif1"
gs2_fexpr0_longname = "gs2_electrostatic_new/beta0_fexpr0"
gs2_fexpr1_longname = "gs2_electrostatic_new/beta0_fexpr1"

def plot_upwinding_scan():
    """ """
    make_comparison_plots([
                    stella_nperiod5_longname,
                    #stella_nperiod5_upwind02_longname,
                    #stella_nperiod5_upwind05_longname,
                    #stella_nperiod5_upwind1_longname,
                    stella_nperiod5_upwind0_longname,
                    stella_nperiod5_t_upwind0_longname,
                    stella_nperiod5_t_upwind0_zvpa_upwind_0_longname,
                    gs2_basecase_longname,
                    gs2_bakdif0_longname,
                    gs2_bakdif1_longname,
                    gs2_fexpr0_longname,
                    gs2_bakdif0_fexpr05_longname
                    #gs2_fexpr1_longname
                    ],
                    [
                    "stella (z, vpa upwind=0.02)",
                    #"stella (z, vpa upwind=0.2)",
                    #"stella (z, vpa upwind=0.5)",
                    #"stella (z, vpa upwind=1.0)",
                    "stella (z, vpa upwind=0.0)",
                    "stella (z, t_upwind=0)",
                    "stella (z, t_upwind=0, vpa upwind=0.0)",
                    "GS2",
                    "GS2 (bakdif=0)",
                    "GS2 (bakdif=1)",
                    "GS2 (fexpr=0)",
                    "GS2 (fexpr=0.5, bakdif=0)",
                    ],
                    "./test_cbc_beta0_upwinding",
                    sim_types=[
                    "stella",
                    #"stella",
                    #"stella",
                    #"stella",
                    "stella",
                    "stella",
                    "stella",
                    "gs2",
                    "gs2",
                    "gs2",
                    "gs2",
                    "gs2",
                    ],
                    plot_format=".png")

    return

def plot_different_mirror_treatment():
    """ """
    make_comparison_plots([
                    stella_nperiod5_longname,
                    stella_nperiod5_t_upwind0_zvpa_upwind_0_longname,
                    stella_nperiod5_t_upwind0_zvpa_upwind_0_implicit_mirror_longname,
                    gs2_basecase_longname,
                    ],
                    [
                    "stella (z_upwind=0.02, t_upwind=)",
                    "stella (z_upwind=, t_upwind=0, SL mirror)",
                    "stella (z_upwind=, t_upwind=0, implicit mirror)",
                    "GS2",
                    ],
                    "./test_cbc_beta0_mirror",
                    sim_types=[
                    "stella",
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".png")

    return

def plot_eqarc_results():
    """ """
    make_comparison_plots([
                    stella_nperiod5_longname,
                    gs2_basecase_longname,
                    gs2_eqarc_false_longname
                    ],
                    [
                    "stella master",
                    "GS2 (equal_arc=.true.)",
                    "GS2 (equal_arc=.false.)",
                    ],
                    "./test_cbc_beta0_eqarc",
                    sim_types=[
                    "stella",
                    "gs2",
                    "gs2",
                    ],
                    plot_format=".png")

    return

def plot_adiabatic_mass_scan():
    """ """
    make_comparison_plots([
                    stella_adiabatic_longname,
                    gs2_adiabatic_longname,
                    stella_adiabatic_m001_longname,
                    gs2_adiabatic_m001_longname,
                    stella_adiabatic_m01_longname,
                    gs2_adiabatic_m01_longname,
                    stella_adiabatic_m05_longname,
                    gs2_adiabatic_m05_longname,
                    stella_adiabatic_m2_longname,
                    gs2_adiabatic_m2_longname,
                    stella_adiabatic_m10_longname,
                    gs2_adiabatic_m10_longname,
                    ],
                    [
                    "stella ",
                    "GS2",
                    "stella (m_i = 0.01)",
                    "GS2 (m_i = 0.01)",
                    "stella (m_i = 0.1)",
                    "GS2 (m_i = 0.1)",
                    "stella (m_i = 0.5)",
                    "GS2 (m_i = 0.5)",
                    "stella (m_i = 2)",
                    "GS2 (m_i = 2)",
                    "stella (m_i = 10)",
                    "GS2 (m_i = 10)",
                    ],
                    "./test_cbc_beta0_adiabatic_mass_scan",
                    sim_types=[
                    "stella",
                    "gs2",
                    "stella",
                    "gs2",
                    "stella",
                    "gs2",
                    "stella",
                    "gs2",
                    "stella",
                    "gs2",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".png")

    return

def plot_adiabatic_results():
    """ """
    make_comparison_plots([
                    stella_adiabatic_longname,
                    gs2_adiabatic_longname,
                    stella_nperiod5_longname,
                    gs2_basecase_longname
                    ],
                    [
                    "stella master (adiabatic electrons)",
                    "GS2 (adiabatic electrons)",
                    "stella master (kinetic electrons)",
                    "GS2 (kinetic electrons)",
                    ],
                    "./test_cbc_beta0_adiabatic",
                    sim_types=[
                    "stella",
                    "gs2",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".png")

    return

def plot_ntheta_scan():
    """ """
    make_comparison_plots([
                    stella_nperiod5_longname,
                    stella_nperiod5_ntheta128_longname,
                    gs2_basecase_longname
                    ],
                    [
                    "stella master (nperiod=5, ntheta=64)",
                    "stella master (nperiod=5, ntheta=128)",
                    "GS2",
                    ],
                    "./test_cbc_beta0_2species_nthetascan",
                    sim_types=[
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".png")
    return

def plot_nperiod_scan():
    """ """
    make_comparison_plots([
                    stella_basecase_longname,
                    stella_nperiod5_longname,
                    stella_nperiod7_longname,
                    stella_nperiod9_longname,
                    gs2_basecase_longname
                    ],
                    [
                    "stella master (nperiod=3)",
                    "stella master (nperiod=5)",
                    "stella master (nperiod=7)",
                    "stella master (nperiod=8)",
                    "GS2 (nperiod=4)",
                    ],
                    "./test_cbc_beta0_2species_nperiodscan",
                    sim_types=[
                    "stella",
                    "stella",
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".png")
    return

def plot_dt_scan():
    """ """
    make_comparison_plots([
                    stella_nperiod5_longname,
                    stella_nperiod5_dt0001_longname,
                    gs2_basecase_longname,
                    gs2_dt001_longname
                    ],
                    [
                    "stella master (dt=0.013)",
                    "stella master (dt=0.01)",
                    "GS2 (dt=0.05)",
                    "GS2 (dt=0.01)",
                    ],
                    "./test_cbc_beta0_2species_dtscan",
                    sim_types=[
                    "stella",
                    "stella",
                    "gs2",
                    "gs2"
                    ],
                    plot_format=".png")
    return

def compare_stella_to_gs2():
    """ """
    master_sim_longname1 = "master_cmiller_es_2species_ypi/cmiller_electrostatic_2species"
    master_sim_longname2 = "master_cmiller_es_2species_marconi/increased_nvpa/input"
    master_sim_longname3 = "master_cmiller_es_2species_marconi/input"
    em_1field_sim_longname = "electromagnetic_1field/cmiller_beta0_2species_explicit_emsields_0"
    #em_3field_sim_longname = "electromagnetic_3fields/cmiller_beta0_2species_explicit"
    gs2_sim_longname = "gs2_electrostatic/cmiller_new_normal_0.0000"
    gs2new_sim_longname = "gs2_electrostatic_new/_0.0000"
    # make_comparison_plots([
    #                 #master_sim_longname1,
    #                 master_sim_longname2,
    #                 master_sim_longname3,
    #                 # em_1field_sim_longname,
    #                 # em_3field_sim_longname,
    #                 gs2_sim_longname,
    #                 gs2new_sim_longname,
    #                 ],
    #                 [
    #                 #"stella master (nvgrid=24, nmu=12, shear=0.8)",
    #                 "stella master (nvgrid=72, nmu=24)",
    #                 "stella master (nvgrid=144, nmu=48)",
    #                 #"feature/electromagnetic, 1 field",
    #                 #"feature/electromagnetic, 3 fields",
    #                 "GS2 (ngauss=8, negrid=16)",
    #                 "GS2 (ngauss=24, negrid=64)",
    #                 ],
    #                 "./test_cbc_beta0_2species",
    #                 sim_types=[
    #                 #"stella",
    #                 #"stella",
    #                 "stella",
    #                 "stella",
    #                 #"stella",
    #                 "gs2",
    #                 "gs2",
    #                 ],
    #                 plot_format=".png")

    # Code to compare geometry between stella and gs2
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    z, gds2, gds21, gds22, bmag, gradpar = extract_data_from_ncdf(master_sim_longname3+".out.nc",
    'zed', 'gds2', 'gds21', 'gds22', 'bmag', 'gradpar')
    ax1.plot(z/np.pi, gds2, label="stella")
    ax2.plot(z/np.pi, gds21, label="stella")
    ax3.plot(z/np.pi, gds22, label="stella")
    ax4.plot(z/np.pi, bmag, label="stella")
    #ax2.plot(z, gradpar)

    theta, gds2, gds21, gds22, bmag, gradpar = extract_data_from_ncdf(gs2new_sim_longname + ".out.nc",
                                    'theta', 'gds2', 'gds21', 'gds22', 'bmag', 'gradpar')
    ax1.plot(theta/np.pi, gds2, linestyle="-.", c="red", ls="..", label="GS2")
    ax2.plot(theta/np.pi, gds21, linestyle="-.", c="red", ls="..", label="GS2")
    ax3.plot(theta/np.pi, gds22, linestyle="-.", c="red", ls="..", label="GS2")
    ax4.plot(theta/np.pi, bmag, linestyle="-.", c="red", ls="..", label="GS2")
    #ax2.plot(theta, gradpar, linestyle="-.", c="red", ls="..")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc="best")
        ax.grid(True)
    ax3.set_xlabel(r"$\theta/\pi$")
    ax4.set_xlabel(r"$\theta/\pi$")
    ax1.set_ylabel("gds2")
    ax2.set_ylabel("gds21")
    ax3.set_ylabel("gds22")
    ax4.set_ylabel("bmag")
    plt.show()

    # view_ncdf_variables(master_sim_longname3+".out.nc")
    # view_ncdf_variables(gs2new_sim_longname + ".out.nc")

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    z, gbdrift, gbdrift0, cvdrift, cvdrift0, gradpar = extract_data_from_ncdf(master_sim_longname3+".out.nc",
    'zed', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0', 'gradpar')
    ax1.plot(z/np.pi, gbdrift, label="stella")
    ax2.plot(z/np.pi, gbdrift0, label="stella")
    ax3.plot(z/np.pi, cvdrift, label="stella")
    ax4.plot(z/np.pi, cvdrift0, label="stella")
    #ax2.plot(z, gradpar)

    theta, gbdrift, gbdrift0, cvdrift, cvdrift0, gradpar = extract_data_from_ncdf(gs2new_sim_longname + ".out.nc",
                                    'theta', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0', 'gradpar')
    ax1.plot(theta/np.pi, gbdrift, linestyle="-.", c="red", ls="..", label="GS2")
    ax2.plot(theta/np.pi, gbdrift0, linestyle="-.", c="red", ls="..", label="GS2")
    ax3.plot(theta/np.pi, cvdrift, linestyle="-.", c="red", ls="..", label="GS2")
    ax4.plot(theta/np.pi, cvdrift0, linestyle="-.", c="red", ls="..", label="GS2")
    #ax2.plot(theta, gradpar, linestyle="-.", c="red", ls="..")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc="best")
        ax.grid(True)
    ax3.set_xlabel(r"$\theta/\pi$")
    ax4.set_xlabel(r"$\theta/\pi$")
    ax1.set_ylabel("gbdrift")
    ax2.set_ylabel("gbdrift0")
    ax3.set_ylabel("cvdrift")
    ax4.set_ylabel("cvdrift0")
    plt.show()


    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.
    return

def plot_g_for_stella_sim():
    """ """
    master_outnc_longname2 = "master_cmiller_es_2species_marconi/input.out.nc"
    #plot_gmvus(master_outnc_longname2, which="gvpa", plot_gauss_squared=True, stretch_electron_vpa=False)
    plot_gzvs(master_outnc_longname2, which="gvpa", plot_gauss_squared=True, stretch_electron_vpa=False)
    return


if __name__ == "__main__":
    #plot_g_for_stella_sim()
    #compare_stella_to_gs2()
    #plot_nperiod_scan()
    #plot_adiabatic_results()
    #plot_dt_scan()
    #plot_eqarc_results()
    #plot_upwinding_scan()
    #plot_adiabatic_mass_scan()
    #plot_ntheta_scan()
    plot_different_mirror_treatment()
