""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots, plot_gmvus, plot_gzvs
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import matplotlib.pyplot as plt
import numpy as np

stella_basecase_longname = "master_cmiller_es_2species_marconi/input"
stella_nperiod5_longname = "master_cmiller_es_2species_marconi/input5"
stella_nperiod5_me00001_longname = "master_cmiller_es_2species_marconi/input5_me0.0001"
stella_nperiod5_me1_longname = "master_cmiller_es_2species_marconi/input5_me1"
stella_nperiod5_rmaj1point1_longname = "master_cmiller_es_2species_marconi/input5_rmaj1.1"
stella_nperiod5_rmaj10_longname = "master_cmiller_es_2species_marconi/input5_rmaj10"
stella_nperiod5_rmaj20_longname = "master_cmiller_es_2species_marconi/input5_rmaj20"
stella_nperiod5_ntheta128_longname = "master_cmiller_es_2species_marconi/input5_ntheta128"
stella_nperiod5_dt0001_longname = "master_cmiller_es_2species_marconi/input5_dt001"
stella_nperiod5_dt00005_longname = "master_cmiller_es_2species_marconi/input5_dt0005"
stella_nperiod5_upwind02_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_0.2"
stella_nperiod5_upwind05_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_0.5"
stella_nperiod5_upwind1_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_1"
stella_nperiod5_upwind0_longname = "master_cmiller_es_2species_marconi/input5_zvpa_upwind_0"
stella_nperiod5_t_upwind0_longname = "master_cmiller_es_2species_marconi/input5_t_upwind0"
stella_nperiod5_t_upwind0_zvpa_upwind_0_longname = "master_cmiller_es_2species_marconi/input5_t_zvpa_upwind0"
stella_nperiod5_t_upwind0_zvpa_upwind_0_implicit_mirror_longname = "master_cmiller_es_2species_marconi/input5_t_zvpa_upwind0_implicit_mirror"
stella_nperiod5_t_upwind0_zvpa_upwind_0_implicit_mirror_flipflop_longname = "master_cmiller_es_2species_marconi/input5_t_zvpa_upwind0_implicit_mirror_flipflop"
stella_nperiod7_longname = "master_cmiller_es_2species_marconi/input7"
stella_nperiod9_longname = "master_cmiller_es_2species_marconi/input9"

# stella, kinetic electrons, from the "new folder" (which has nperiod=5 and no upwinding)
stella_noupwind_emirror_estream_edrift_higher_vres = "master_cmiller_noupwind_2species_marconi/input_emirror_estream_edrift_higher_vres"
stella_noupwind_imirror_istream_edrift_flipflop = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift_flipflop"
stella_noupwind_imirror_istream_edrift_me1 = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift_me1"
stella_noupwind_imirror_istream_idrift = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_idrift"
stella_noupwind_imirror_istream_idrift_dt001 = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_idrift_dt001"
stella_noupwind_imirror_istream_idrift_dt0005 = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_idrift_dt0005"
stella_noupwind_emirror_estream_edrift = "master_cmiller_noupwind_2species_marconi/input_emirror_estream_edrift"
stella_noupwind_imirror_istream_edrift_higher_vres = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift_higher_vres"
stella_noupwind_imirror_istream_edrift_midvres = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift_midvres"
stella_noupwind_imirror_istream_edrift_smaller_dt = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift_smaller_dt"
stella_noupwind_slmirror_istream_edrift_higher_vres = "master_cmiller_noupwind_2species_marconi/input_slmirror_istream_edrift_higher_vres"
stella_noupwind_emirror_estream_edrift_smaller_dt = "master_cmiller_noupwind_2species_marconi/input_emirror_estream_edrift_smaller_dt"
stella_noupwind_imirror_istream_edrift = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift"
stella_noupwind_imirror_istream_edrift_stm1 = "master_cmiller_noupwind_2species_marconi/input_imirror_istream_edrift_stm1"
stella_noupwind_slmirror_istream_edrift = "master_cmiller_noupwind_2species_marconi/input_slmirror_istream_edrift"
stella_debug_flipflop = "master_cmiller_noupwind_2species_marconi/iie_ff_from_restart"


# Adiabatic stella
stella_adiabatic_longname = "stella_adiabatic/input5"
stella_adiabatic_m001_longname = "stella_adiabatic/input5_mi0.01"
stella_adiabatic_m01_longname = "stella_adiabatic/input5_mi0.1"
stella_adiabatic_m05_longname = "stella_adiabatic/input5_mi0.5"
stella_adiabatic_m2_longname = "stella_adiabatic/input5_mi2"
stella_adiabatic_m10_longname = "stella_adiabatic/input5_mi10"

# Adiabatic stella, leapfrog drifts
stella_adiabatic_leapfrog_drifts_longname = "stella_adiabatic_leapfrog_drifts/input5"
stella_adiabatic_m001_leapfrog_drifts_longname = "stella_adiabatic_leapfrog_drifts/input5_mi0.01"
stella_adiabatic_m01_leapfrog_drifts_longname = "stella_adiabatic_leapfrog_drifts/input5_mi0.1"
stella_adiabatic_m05_leapfrog_drifts_longname = "stella_adiabatic_leapfrog_drifts/input5_mi0.5"
stella_adiabatic_m2_leapfrog_drifts_longname = "stella_adiabatic_leapfrog_drifts/input5_mi2"
stella_adiabatic_m10_leapfrog_drifts_longname = "stella_adiabatic_leapfrog_drifts/input5_mi10"

# stella 2 species, leapfrog drifts
stella_nperiod5_leapfrog_drifts_longname = "stella_cmiller_es_2species_leapfrog_drifts/input5"
stella_nperiod3_leapfrog_drifts_longname = "stella_cmiller_es_2species_leapfrog_drifts/input"
stella_dt0005_leapfrog_drifts_longname = "stella_cmiller_es_2species_leapfrog_drifts/input5_dt0005"

gs2_basecase_longname = "gs2_electrostatic_new/_0.0000"
gs2_rmaj10_longname = "gs2_electrostatic_new/beta0_bakdif0_fexpr0.5_rmaj10"
gs2_rmaj20_longname = "gs2_electrostatic_new/beta0_bakdif0_fexpr0.5_rmaj20"
gs2_rmaj1point1_longname = "gs2_electrostatic_new/beta0_bakdif0_fexpr0.5_rmaj1.1"
gs2_me1_longname = "gs2_electrostatic_new/beta0_bakdif0_fexpr0.5_me1"
gs2_nperiod6_longname = "gs2_electrostatic_new/beta0_nperiod6"
gs2_ntheta128_longname = "gs2_electrostatic_new/beta0_ntheta128"
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
                    gs2_basecase_longname,
                    gs2_ntheta128_longname,

                    ],
                    [
                    "stella master (nperiod=5, ntheta=64)",
                    "stella master (nperiod=5, ntheta=128)",
                    "GS2 (nperiod=4, ntheta=64)",
                    "GS2 (nperiod=4, ntheta=128)",
                    ],
                    "./test_cbc_beta0_2species_nthetascan",
                    sim_types=[
                    "stella",
                    "stella",
                    "gs2",
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
                    gs2_basecase_longname,
                    gs2_nperiod6_longname
                    ],
                    [
                    "stella master (nperiod=3)",
                    "stella master (nperiod=5)",
                    "stella master (nperiod=7)",
                    "stella master (nperiod=8)",
                    "GS2 (nperiod=4)",
                    "GS2 (nperiod=6)",
                    ],
                    "./test_cbc_beta0_2species_nperiodscan",
                    sim_types=[
                    "stella",
                    "stella",
                    "stella",
                    "stella",
                    "gs2",
                    "gs2",
                    ],
                    plot_format=".png")
    return

def plot_dt_scan():
    """ """
    make_comparison_plots([
                    stella_nperiod5_longname,
                    stella_nperiod5_dt0001_longname,
                    stella_nperiod5_dt00005_longname,
                    gs2_basecase_longname,
                    gs2_dt001_longname
                    ],
                    [
                    "stella master (dt=0.013)",
                    "stella master (dt=0.01)",
                    "stella master (dt=0.005)",
                    "GS2 (dt=0.05)",
                    "GS2 (dt=0.01)",
                    ],
                    "./test_cbc_beta0_2species_dtscan",
                    sim_types=[
                    "stella",
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

    # # Code to compare geometry between stella and gs2
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
    #sys.exit()
    #
    # # view_ncdf_variables(master_sim_longname3+".out.nc")
    # # view_ncdf_variables(gs2new_sim_longname + ".out.nc")
    #
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
    sys.exit()
    #

    ## Try plotting the drifts
    # GS2 has (see Highcock thesis) vdrift . grad h = i * (T/Z) * omega_drift h
    # omega_drift = (ky/2) (vperp^2/2 * (gbdrift + kx/(ky*shat)gbdrift0)
    #                        + vpa^2 * (cvdrift + kx/(ky*shat)cvdrift0))
    #
    # stella has (barnes 2019) vdrift . grad h = i * (T/(Z*B)) * omega_drift h
    # omega_drift = (vpa^2 * v_k + mu * v_gradb)(ky grad y  +kx grad x)

    z, gbdrift, gbdrift0, cvdrift, cvdrift0, gradpar, vpa, mu = extract_data_from_ncdf(master_sim_longname3+".out.nc",
    'zed', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0', 'gradpar', 'vpa', 'mu')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for mu_idx in range(0, len(mu)):
        single_mu = mu[mu_idx]*np.ones(len(vpa))
        ax1.scatter(vpa, np.sqrt(single_mu), marker="x", c="black")
    ax1.set_xlabel("vpa")
    ax1.set_ylabel("mu")
    plt.show()


    # view_ncdf_variables(gs2new_sim_longname+".out.nc")
    return

def plot_g_for_stella_sim():
    """ """
    master_outnc_longname2 = "master_cmiller_es_2species_marconi/input.out.nc"
    #plot_gmvus(master_outnc_longname2, which="gvpa", plot_gauss_squared=True, stretch_electron_vpa=False)
    plot_gzvs(master_outnc_longname2, which="gvpa", plot_gauss_squared=True, stretch_electron_vpa=False)
    return

def plot_flip_flop_option():
    """ """
    make_comparison_plots([
            stella_nperiod5_longname,
            stella_nperiod5_t_upwind0_zvpa_upwind_0_implicit_mirror_longname,
            stella_nperiod5_t_upwind0_zvpa_upwind_0_implicit_mirror_flipflop_longname,
            gs2_basecase_longname
                    ],
                    [
                    "stella base case",
                    "stella upwinding=0, implicit mirror",
                    "stella upwinding=0, implicit mirror, flip-flop",
                    "GS2",
                    ],
                    "./test_cbc_beta0_2species_flipflop",
                    sim_types=[
                    "stella",
                    "stella",
                    "stella",
                    "gs2"
                    ],
                    plot_format=".png")

def plot_rmaj_scan():
    """ """
    make_comparison_plots([
            stella_nperiod5_longname,
            stella_nperiod5_rmaj1point1_longname,
            #stella_nperiod5_rmaj10_longname,
            stella_nperiod5_rmaj20_longname,
            gs2_basecase_longname,
            gs2_rmaj1point1_longname,
            gs2_rmaj20_longname,
                    ],
                    [
                    "stella base case",
                    "stella rmaj=1.1",
                    #"stella rmaj=10",
                    "stella rmaj=20",
                    "GS2",
                    "GS2 rmaj=1.1",
                    "GS2 rmaj=20",
                    "GS2",
                    ],
                    "./test_cbc_beta0_2species_rmajscan",
                    sim_types=[
                    "stella",
                    "stella",
                    #"stella",
                    "stella",
                    "gs2",
                    "gs2",
                    "gs2",
                    ],
                    plot_format=".png")

def plot_stella_gs2_nperiod5():
    """ """
    make_comparison_plots([
            stella_nperiod5_longname,
            gs2_basecase_longname,
                    ],
                    [
                    "stella",
                    "GS2",
                    ],
                    "./test_cbc_beta0_2species_default_comparison",
                    sim_types=[
                    "stella",
                    "gs2",
                    ],
                    plot_format=".png")


def plot_me_scan():
    """ """
    make_comparison_plots([
            stella_nperiod5_longname,
            #stella_nperiod5_me00001_longname,
            stella_nperiod5_me1_longname,
            gs2_basecase_longname,
            gs2_me1_longname
                    ],
                    [
                    "stella base case (me=2.8e-4)",
                    #"stella me=1e-4",
                    "stella me=1",
                    "GS2 (me=2.8e-4)",
                    "GS2 (me=1)",
                    ],
                    "./test_cbc_beta0_2species_mescan",
                    sim_types=[
                    "stella",
                    #"stella",
                    #"stella",
                    "stella",
                    "gs2",
                    "gs2"
                    ],
                    plot_format=".png", show_fig=True)


def plot_noupwind_different_numerical_schemes():
    """With no upwidning, see what happens when we change the numerical scheme,
    and the v-space resolution """

    make_comparison_plots([
                stella_noupwind_imirror_istream_edrift,
                stella_noupwind_imirror_istream_edrift_higher_vres,
                stella_noupwind_imirror_istream_edrift_midvres,
                stella_noupwind_slmirror_istream_edrift,
                stella_noupwind_slmirror_istream_edrift_higher_vres,
                stella_noupwind_emirror_estream_edrift,
                stella_noupwind_emirror_estream_edrift_higher_vres,
                stella_noupwind_imirror_istream_idrift,
                gs2_basecase_longname,
                gs2_bakdif0_fexpr05_longname                                ],
                                [
                                "i, i, e std. vres",
                                "i, i, e, higher_vres",
                                "i, i, e, mid_vres",
                                "sl, i, e std. vres",
                                "sl, i, e, higher_vres",
                                "e, e, e std. vres",
                                "e, e, e, higher_vres",
                                "i, i, i std. vres",
                                "GS2 (default)",
                                "GS2 (bakdif=0, fexpr=0.5)",
                                ],
                    "images/noupwind_vres_scan",
                    sim_types=[
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "gs2",
                        "gs2",
                        ],
                    plot_format=".png", show_fig=True)
    return

def plot_noupwind_dt_variation():
    """See if we're resolved in dt for each scheme"""
    make_comparison_plots([
            stella_noupwind_imirror_istream_edrift,
            stella_noupwind_imirror_istream_edrift_smaller_dt,
            stella_noupwind_emirror_estream_edrift,
            stella_noupwind_emirror_estream_edrift_smaller_dt,
            stella_noupwind_imirror_istream_idrift,
            stella_noupwind_imirror_istream_idrift_dt001,
            stella_noupwind_imirror_istream_idrift_dt0005,
            gs2_basecase_longname,
            gs2_bakdif0_fexpr05_longname
            ],
            [
                "i, i, e, dt=0.0133",
                "i, i, e, dt=0.005",
                "e, e, e, dt=0.0133",
                "e, e, e, dt=0.005",
                "i, i, i, dt=0.03",
                "i, i, i, dt=0.01",
                "i, i, i, dt=0.005",
                "GS2 (default)",
                "GS2 (bakdif=0, fexpr=0.5)",
            ],
            "images/noupwind_dt_variation",
            sim_types = [
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "stella",
                        "gs2",
                        "gs2",
            ],
                plot_format=".png", show_fig=True)
    return


def plot_noupwind_different_electron_treatment():
    """See what happens if we (1)  """
    make_comparison_plots([
            stella_noupwind_imirror_istream_edrift,
            stella_noupwind_imirror_istream_edrift_me1,
            stella_noupwind_imirror_istream_edrift_stm1,
            gs2_basecase_longname
                    ],
                    [
                    "i, i, e",
                    "i, i, e, me=1",
                    "i, i, e, stm=1",
                    "GS2",
                    "GS2, me=1",
                    "GS2, stm=1",
                    ],
                    "images/noupwind_diff_electron_treatment",
                    sim_types=[
                        "stella",
                        "stella",
                        "stella",
                        "gs2",
                        "gs2",
                        "gs2",
                    ],
                plot_format=".png")
    return

def plot_noupwind_flipflop():
    """Compare a case with flip_flop on vs flip_flop off"""
    make_comparison_plots([
            stella_noupwind_imirror_istream_edrift,
            stella_noupwind_imirror_istream_edrift_flipflop,
            gs2_basecase_longname
                ],
                [
                 "i, i, e",
                 "i, i, e, flip-flop",
                 "GS2",
                ],
                "images/noupwind_flipflop",
                sim_types=[
                        "stella",
                        "stella",
                        "gs2"
                ],
            plot_format=".png")
    return

def plot_phit_noupwind_flip_flop():
    """ """
    stella_noupwind_imirror_istream_edrift_outnc = stella_noupwind_imirror_istream_edrift + ".out.nc"
    stella_noupwind_imirror_istream_edrift_flipflop_outnc = stella_noupwind_imirror_istream_edrift_flipflop + ".out.nc"

    [t, z, lie_phi_vs_t] = extract_data_from_ncdf(stella_noupwind_imirror_istream_edrift_outnc, "t", "zed", "phi_vs_t")
    [t, z, flipflop_phi_vs_t] = extract_data_from_ncdf(stella_noupwind_imirror_istream_edrift_flipflop_outnc, "t", "zed", "phi_vs_t")

    print("len(t) = ", len(t))
    print("len(z) = ", len(z))
    #print("phi_vs_t.shape = ", phi_vs_t.shape)  # (t, tube, z, ky, kx) I think
    lie_phi_avg = []
    flipflop_phi_avg = []

    for t_idx in range(0,len(t)):
        lie_phi_vs_t_idx = lie_phi_vs_t[t_idx,0,:,0,0]
        flipflop_phi_vs_t_idx = flipflop_phi_vs_t[t_idx,0,:,0,0]

        lie_phi_avg.append(np.mean(lie_phi_vs_t_idx))
        flipflop_phi_avg.append(np.mean(flipflop_phi_vs_t_idx))

    lie_phi_avg = np.array(lie_phi_avg)
    flipflop_phi_avg = np.array(flipflop_phi_avg)
    lie_omega = np.log(lie_phi_avg[1:]/lie_phi_avg[:-1]) *1j
    flipflop_omega = np.log(flipflop_phi_avg[1:]/flipflop_phi_avg[:-1]) *1j
    print("lie_omega = ", lie_omega)
    print("flipflop_omega = ", flipflop_omega)
    # for t_idx in range(0,10):
    #     lie_phi_vs_t_idx = lie_phi_vs_t[t_idx,0,:,0,0]
    #     flipflop_phi_vs_t_idx = flipflop_phi_vs_t[t_idx,0,:,0,0]
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.plot(z, abs(lie_phi_vs_t_idx))
    #     ax1.plot(z, abs(flipflop_phi_vs_t_idx))
    #     plt.show()

    return

def debug_flip_flop():
    """ """
    stella_debug_flipflop_outnc = stella_debug_flipflop + ".out.nc"

    view_ncdf_variables(stella_debug_flipflop_outnc)
    # ['code_info', 'nproc', 'nmesh', 'ntubes', 'nkx', 'nky', 'nzed_tot',
    #  'nspecies', 'nmu', 'nvpa_tot', 't', 'charge', 'mass', 'dens', 'temp',
    #  'tprim', 'fprim', 'vnew', 'type_of_species', 'theta0', 'kx', 'ky', 'mu',
    #  'vpa', 'zed', 'bmag', 'gradpar', 'gbdrift', 'gbdrift0', 'cvdrift',
    #  'cvdrift0', 'kperp2', 'gds2', 'gds21', 'gds22', 'grho', 'jacob', 'q',
    #  'beta', 'shat', 'jtwist', 'drhodpsi', 'phi2', 'phi_vs_t', 'gvmus', 'gzvs', 'input_file']

    [input_file, t, z, flipflop_phi_vs_t] = extract_data_from_ncdf(stella_debug_flipflop_outnc, "input_file", "t", "zed", "phi_vs_t")
    #print("code_info = ", code_info) # [-- -- -- -- -- -- -- -- -- --]
    #sys.exit()
    print("len(t) = ", len(t))
    print("len(z) = ", len(z))
    print("input_file = ", input_file)
    #print("phi_vs_t.shape = ", phi_vs_t.shape)  # (t, tube, z, ky, kx) I think
    flipflop_phi_avg = []
    code_dt = 0.013
    for t_idx in range(0,len(t)):
        flipflop_phi_vs_t_idx = flipflop_phi_vs_t[t_idx,0,:,0,0]

        #flipflop_phi_avg.append(np.mean(flipflop_phi_vs_t_idx))
        flipflop_phi_avg.append(flipflop_phi_vs_t_idx[int(len(z)*0.5)])
    flipflop_phi_avg = np.array(flipflop_phi_avg)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    flip_flop_diff_1 = np.abs(flipflop_phi_vs_t[1,0,:,0,0]) - np.abs(flipflop_phi_vs_t[0,0,:,0,0])
    flip_flop_diff_2 = np.abs(flipflop_phi_vs_t[2,0,:,0,0]) - np.abs(flipflop_phi_vs_t[1,0,:,0,0])
    flip_flop_diff_3 = np.abs(flipflop_phi_vs_t[2,0,:,0,0]) - np.abs(flipflop_phi_vs_t[0,0,:,0,0])

    #flip_flop_diff_4 = np.abs(flipflop_phi_vs_t[4,0,:,0,0]) - np.abs(flipflop_phi_vs_t[2,0,:,0,0])
    #print("flip_flop_diff = ", flip_flop_diff)
    ax1.plot(z/np.pi, flip_flop_diff_1, label="abs(phi(t1)) - abs(phi(t0))")
    ax1.plot(z/np.pi, flip_flop_diff_2, label="abs(phi(t2)) - abs(phi(t1))")
    ax1.plot(z/np.pi, flip_flop_diff_3, label="abs(phi(t2)) - abs(phi(t0))")
    ax1.plot(z/np.pi, np.abs(flipflop_phi_vs_t[2,0,:,0,0]), label="abs(phi(t2))")
    ax1.legend(loc="best")
    ax1.grid(True)
    ax1.set_xlabel(r"$z/\pi$")
    #ax1.plot(z, flip_flop_diff_4)
    #ax1.plot(t, abs(flipflop_phi_avg))
    plt.show()
    flipflop_omega = np.log(flipflop_phi_avg[1:]/flipflop_phi_avg[:-1]) *1j/code_dt
    print("flipflop_omega = ", flipflop_omega)
    # for t_idx in range(0,10):
    #     lie_phi_vs_t_idx = lie_phi_vs_t[t_idx,0,:,0,0]
    #     flipflop_phi_vs_t_idx = flipflop_phi_vs_t[t_idx,0,:,0,0]
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.plot(z, abs(lie_phi_vs_t_idx))
    #     ax1.plot(z, abs(flipflop_phi_vs_t_idx))
    #     plt.show()

    return

def compare_stella_leapfrog_gs2():
    """ """
    make_comparison_plots([
            stella_nperiod5_longname,
            stella_nperiod5_leapfrog_drifts_longname,
            gs2_basecase_longname,
                    ],
                    [
                    "stella (Lie)",
                    "stella (Leapfrog drifts)",
                    "GS2",
                    ],
                    "./test_cbc_beta0_2species_stella_leapfrog",
                    sim_types=[
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".eps")

    make_comparison_plots([
            stella_nperiod5_longname,
            stella_nperiod3_leapfrog_drifts_longname,
            stella_nperiod5_leapfrog_drifts_longname,
            stella_dt0005_leapfrog_drifts_longname,
            gs2_basecase_longname,
                    ],
                    [
                    "stella (Lie)",
                    "stella (Leapfrog drifts)",
                    "stella (Leapfrog drifts, np=3)",
                    "stella (Leapfrog drifts, dt=5E-3)",
                    "GS2",
                    ],
                    "./test_cbc_beta0_2species_stella_leapfrog",
                    sim_types=[
                    "stella",
                    "stella",
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".eps")
    return

def compare_stella_leapfrog_gs2_adiabatic():
    """ """

    make_comparison_plots([
                stella_adiabatic_longname,
                stella_adiabatic_leapfrog_drifts_longname,
                gs2_adiabatic_longname,
                    ],
                    [
                    "stella (Lie)",
                    "stella (Leapfrog drifts)",
                    "GS2",
                    ],
                    "./test_cbc_beta0_adiabatic_stella_leapfrog",
                    sim_types=[
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".eps")

    make_comparison_plots([
                stella_adiabatic_m01_longname,
                stella_adiabatic_m01_leapfrog_drifts_longname,
                gs2_adiabatic_m001_longname,
                stella_adiabatic_m10_longname,
                stella_adiabatic_m10_leapfrog_drifts_longname,
                gs2_adiabatic_m10_longname
                    ],
                    [
                    "stella m=0.1 (Lie)",
                    "stella m=0.1 (Leapfrog drifts)",
                    "GS2 m=0.1",
                    "stella m=10 (Lie)",
                    "stella m=10 (Leapfrog drifts)",
                    "GS2 m=10",
                    ],
                    "./test_cbc_beta0_adiabatic_stella_leapfrog_mass_scan",
                    sim_types=[
                    "stella",
                    "stella",
                    "gs2",
                    "stella",
                    "stella",
                    "gs2",
                    ],
                    plot_format=".eps")

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
    #plot_different_mirror_treatment()
    #plot_flip_flop_option()
    #plot_me_scan()
    #plot_rmaj_scan()
    #compare_stella_to_gs2()
    # plot_noupwind_different_numerical_schemes()
    # plot_noupwind_dt_variation()
    # plot_phit_noupwind_flip_flop()
    compare_stella_leapfrog_gs2()
    compare_stella_leapfrog_gs2_adiabatic()
    #debug_flip_flop()
    #plot_noupwind_flipflop()
    #plot_stella_gs2_nperiod5()
