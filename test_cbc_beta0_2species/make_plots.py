""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots, plot_gmvus
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import matplotlib.pyplot as plt
import numpy as np

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
    plot_gmvus(master_outnc_longname2, which="gvpa", plot_gauss_squared=True, stretch_electron_vpa=False)
    return


if __name__ == "__main__":
    plot_g_for_stella_sim()
    #compare_stella_to_gs2()
