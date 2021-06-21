""" """

from extract_sim_data import get_omega_data, get_phiz_data, get_aparz_data, get_bparz_data
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf, extract_data_from_ncdf_with_xarray
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import sys
import pickle

## Define linestyles
linestyles1=cycle(["-", "--", "-.", ":"])
linestyles2=cycle(["-", "--", "-.", ":"])




def plot_omega_t_for_sim(ax1, ax2, sim_longname, sim_label, sim_type="stella"):
    """ """
    time, freqom_final, gammaom_final, freqom, gammaom = get_omega_data(sim_longname, sim_type)
    print("freqom_final, gammaom_final = ", freqom_final, gammaom_final)
    half_len = int(len(time)/2)
    freqom_half = freqom[half_len:]
    gammaom_half = gammaom[half_len:]
    # Get an estimate for the ylims by looking at the max/min of the second half
    # of the frequencuy and gamma data
    if np.isfinite(freqom_final) and np.isfinite(gammaom_final):
        linestyle=next(linestyles1)
        gamma_llim = (np.min(gammaom_half)); gamma_ulim = (np.max(gammaom_half))
        freq_llim = (np.min(freqom_half)); freq_ulim = np.max(freqom_half)
        ax1.plot(time, freqom, label=sim_label, ls=linestyle)
        ax2.plot(time, gammaom, label=sim_label, ls=linestyle)
    else:
        gamma_llim = np.NaN; gamma_ulim = np.NaN; freq_llim = np.NaN; freq_ulim = np.NaN

    return gammaom_final, freqom_final, gamma_llim, gamma_ulim, freq_llim, freq_ulim

def plot_phi_z_for_sim(ax1, sim_longname, sim_label, sim_type="stella", plot_format=".eps"):
    """ """

    theta, real_phi, imag_phi = get_phiz_data(sim_longname, sim_type)
    ## Check values are finite
    if not(np.all(np.isfinite(real_phi)) and np.all(np.isfinite(imag_phi))):
        print("Error! phi contains non-finite values")
        print("sim_longname = ", sim_longname)
        return

    ## Combine real and imaginary parts to get abs_phi
    # If real and imaginary parts are large, it's possible that they'll
    # become non-finite when we square them. To avoid, perform some normalisation first
    normalisation = np.max(abs(real_phi))
    real_phi = real_phi/normalisation
    imag_phi = imag_phi/normalisation
    abs_phi = np.sqrt(real_phi*real_phi + imag_phi*imag_phi)

    ## Normalise s.t. max(abs_phi) = 1
    abs_phi = abs_phi/np.max(abs_phi)
    # Plot
    linestyle=next(linestyles2)
    ax1.plot(theta/np.pi, abs_phi, label=sim_label, ls=linestyle)

    return

def plot_apar_z_for_sim(ax1, sim_longname, sim_label, sim_type="stella"):
    """ """

    theta, real_apar, imag_apar = get_aparz_data(sim_longname, sim_type)
    ## Check values are finite
    if not(np.all(np.isfinite(real_apar)) and np.all(np.isfinite(imag_apar))):
        print("Error! apar contains non-finite values")
        print("sim_longname = ", sim_longname)
        return

    ## Combine real and imaginary parts to get abs_apar
    # If real and imaginary parts are large, it's possible that they'll
    # become non-finite when we square them. To avoid, perform some normalisation first
    normalisation = np.max(abs(real_apar))
    real_apar = real_apar/normalisation
    imag_apar = imag_apar/normalisation
    abs_apar = np.sqrt(real_apar*real_apar + imag_apar*imag_apar)

    ## Normalise s.t. max(abs_apar) = 1
    abs_apar = abs_apar/np.max(abs_apar)
    # Plot
    linestyle=next(linestyles2)
    ax1.plot(theta/np.pi, abs_apar, label=sim_label, ls=linestyle)

    return

def plot_bpar_z_for_sim(ax1, sim_longname, sim_label, sim_type="stella"):
    """ """

    theta, real_bpar, imag_bpar = get_bparz_data(sim_longname, sim_type)
    ## Check values are finite
    if not(np.all(np.isfinite(real_bpar)) and np.all(np.isfinite(imag_bpar))):
        print("Error! bpar contains non-finite values")
        print("sim_longname = ", sim_longname)
        return

    ## Combine real and imaginary parts to get abs_bpar
    # If real and imaginary parts are large, it's possible that they'll
    # become non-finite when we square them. To avoid, perform some normalisation first
    normalisation = np.max(abs(real_bpar))
    real_bpar = real_bpar/normalisation
    imag_bpar = imag_bpar/normalisation
    abs_bpar = np.sqrt(real_bpar*real_bpar + imag_bpar*imag_bpar)

    ## Normalise s.t. max(abs_bpar) = 1
    abs_bpar = abs_bpar/np.max(abs_bpar)
    # Plot
    linestyle=next(linestyles2)
    ax1.plot(theta/np.pi, abs_bpar, label=sim_label, ls=linestyle)

    return

def make_comparison_plots(sim_longnames, sim_labels, save_name, sim_types=[],
                          plot_apar=False, plot_bpar=False, plot_format=".eps"):
    """Compare multiple simulations which have a single common input. Create the following
    plots:
    1) omega(t)
    2) Normalised |phi|(z)
    """
    ## Plot of omega(t)
    fig1 = plt.figure(figsize=[10, 12])
    ax11 = fig1.add_subplot(211)
    ax12 = fig1.add_subplot(212, sharex=ax11)

    ## Plot of |phi|(t)
    fig2 = plt.figure(figsize=[8, 8])
    ax21 = fig2.add_subplot(111)
    if plot_apar:
        fig3 = plt.figure(figsize=[8, 8])
        ax31 = fig3.add_subplot(111)
    if plot_bpar:
        fig4 = plt.figure(figsize=[8, 8])
        ax41 = fig4.add_subplot(111)
    gamma_vals = []
    freq_vals = []

    gamma_llims = []
    gamma_ulims = []
    freq_llims = []
    freq_ulims = []

    for sim_idx, sim_longname in enumerate(sim_longnames):
        sim_label = sim_labels[sim_idx]
        # Find out the sim types - if sim_types kwarg not specified,
        # assume all stella
        if len(sim_types) == 0:
            sim_type="stella"
        elif len(sim_types) == len(sim_longnames):
            sim_type = sim_types[sim_idx]
        else:
            print("Error! len(sim_longnames), len(sim_types) = ", len(sim_longnames), len(sim_types) )
            sys.exit()
        gammaom_final, freqom_final, gamma_llim, gamma_ulim, \
            freq_llim, freq_ulim = plot_omega_t_for_sim(ax11, ax12, sim_longname, sim_label, sim_type=sim_type)
        plot_phi_z_for_sim(ax21, sim_longname, sim_label, sim_type=sim_type)
        if plot_apar:
            plot_apar_z_for_sim(ax31, sim_longname, sim_label, sim_type=sim_type)
        if plot_bpar:
            plot_bpar_z_for_sim(ax41, sim_longname, sim_label, sim_type=sim_type)

        if (np.isfinite(gammaom_final) and np.isfinite(freqom_final)
                and np.isfinite(gamma_llim) and np.isfinite(gamma_ulim)
                and np.isfinite(freq_llim) and np.isfinite(freq_ulim) ):
            gamma_llims.append(gamma_llim)
            gamma_ulims.append(gamma_ulim)
            freq_llims.append(freq_llim)
            freq_ulims.append(freq_ulim)
            gamma_vals.append(gammaom_final)
            freq_vals.append(freqom_final)

    ## Set lims based on sim data
    gamma_llim = np.min(np.array(gamma_llims))/1.1
    gamma_ulim = np.max(np.array(gamma_ulims))*1.1
    freq_llim = np.min(np.array(freq_llims))/1.1
    freq_ulim = np.max(np.array(freq_ulims))*1.1
    ax11.set_ylim(freq_llim, freq_ulim)
    ax12.set_ylim(gamma_llim, gamma_ulim)
    ax21.set_ylim(-0.05, 1.05)
    ax12.set_xlabel(r"$t$")
    ax11.set_ylabel(r"$\omega$")
    ax12.set_ylabel(r"$\gamma$")
    ax21.set_xlabel(r"$\theta/\pi$")
    ax21.set_ylabel(r"$\vert \phi \vert$")
    axes = [ax11, ax12, ax21]
    if plot_apar:
        axes.append(ax31)
        ax31.set_ylim(-0.05, 1.05)
        ax31.set_xlabel(r"$\theta/\pi$")
        ax31.set_ylabel(r"$\vert A_\parallel \vert$")
    if plot_bpar:
        axes.append(ax41)
        ax41.set_ylim(-0.05, 1.05)
        ax41.set_xlabel(r"$\theta/\pi$")
        ax41.set_ylabel(r"$\vert B_\parallel \vert$")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best")
    fig1.savefig(save_name + "_omega" + plot_format)
    fig2.savefig(save_name + "_phi" + plot_format)
    if plot_apar:
        fig3.savefig(save_name + "_apar" + plot_format)
        plt.close(fig3)
    if plot_bpar:
        fig4.savefig(save_name + "_bpar" + plot_format)
        plt.close(fig4)
    plt.close(fig1)
    plt.close(fig2)
    return

def make_beta_scan_plots(sim_longnames, beta_vals, save_name, sim_types=[],
                         gs2_pickle=None):
    """ """
    ## Plot of omega(t)
    fig1 = plt.figure(figsize=[10, 12])
    ax11 = fig1.add_subplot(211)
    ax12 = fig1.add_subplot(212, sharex=ax11)

    ## Plot of omega(beta)
    fig2 = plt.figure(figsize=[10, 12])
    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212, sharex=ax21)
    gamma_vals = []
    freq_vals = []
    final_beta_vals = []
    gamma_llims = []
    gamma_ulims = []
    freq_llims = []
    freq_ulims = []

    gamma_vals = []
    freq_vals = []

    for sim_idx, sim_longname in enumerate(sim_longnames):
        sim_label = "beta = " + str(beta_vals[sim_idx])
        if len(sim_types) == 0:
            sim_type="stella"
        elif len(sim_types) == len(sim_longnames):
            sim_type = sim_types[sim_idx]
        else:
            print("Error! len(sim_longnames), len(sim_types) = ", len(sim_longnames), len(sim_types) )
            sys.exit()
        gammaom_final, freqom_final, gamma_llim, gamma_ulim, \
            freq_llim, freq_ulim = plot_omega_t_for_sim(ax11, ax12, sim_longname, sim_label, sim_type=sim_type)
        if (np.isfinite(gammaom_final) and np.isfinite(freqom_final)
                and np.isfinite(gamma_llim) and np.isfinite(gamma_ulim)
                and np.isfinite(freq_llim) and np.isfinite(freq_ulim) ):
            gamma_llims.append(gamma_llim)
            gamma_ulims.append(gamma_ulim)
            freq_llims.append(freq_llim)
            freq_ulims.append(freq_ulim)
            gamma_vals.append(gammaom_final)
            freq_vals.append(freqom_final)
            final_beta_vals.append(beta_vals[sim_idx])

    ## Set lims based on sim data
    gamma_llim = np.min(np.array(gamma_llims))
    gamma_ulim = np.max(np.array(gamma_ulims))
    freq_llim = np.min(np.array(freq_llims))
    freq_ulim = np.max(np.array(freq_ulims))
    ax11.set_ylim(freq_llim, freq_ulim)
    ax12.set_ylim(gamma_llim, gamma_ulim)
    ax12.set_xlabel(r"$t$")
    ax11.set_ylabel(r"$\omega$")
    ax12.set_ylabel(r"$\gamma$")


    ax21.plot(final_beta_vals, freq_vals, label="stella")
    ax22.plot(final_beta_vals, gamma_vals, label="stella")
    ax21.scatter(final_beta_vals, freq_vals, c="black", s=15., marker="x")
    ax22.scatter(final_beta_vals, gamma_vals, c="black", s=15., marker="x")
    if gs2_pickle is not None:
        myfile = open(gs2_pickle, 'rb')
        gs2_dict = pickle.load(myfile)  # contains 'beta', 'frequency', 'growth rate'
        myfile.close()
        gs2_beta = gs2_dict['beta']; gs2_frequency=gs2_dict['frequency']; gs2_growth_rate = gs2_dict['growth rate']
        # Sort the beta vals into ascending order
        gs2_beta = np.array(gs2_beta).flatten()
        sort_idxs = np.argsort(gs2_beta)
        gs2_beta = np.array(gs2_beta)[sort_idxs]
        gs2_frequency = np.array(gs2_frequency)[sort_idxs]
        gs2_growth_rate = np.array(gs2_growth_rate)[sort_idxs]
        ax21.plot(gs2_beta, gs2_frequency, label="GS2")
        ax22.plot(gs2_beta, gs2_growth_rate, label="GS2")

    ax21.set_ylabel(r"$\omega$")
    ax22.set_ylabel(r"$\gamma$")
    ax22.set_xlabel(r"$\beta$")

    for ax in [ax11, ax12, ax21, ax22]:
        ax.grid(True)
        ax.legend(loc="best")

    fig1.savefig(save_name + "_omega_t.eps")
    fig2.savefig(save_name + "_omega_beta.eps")
    plt.close(fig1)
    plt.close(fig2)

    return


def plot_gmvus(stella_outnc_longname, which="gvpa", plot_gauss_squared=False,
                stretch_electron_vpa=False):
    """ """
    t, z, mu, vpa, gds2, gds21, gds22, bmag, gradpar, gvmus = extract_data_from_ncdf(stella_outnc_longname,
                                    "t", 'zed', "mu", "vpa", 'gds2', 'gds21', 'gds22', 'bmag', 'gradpar', 'gvmus')
    print("len(t)", "len(z), len(mu), len(vpa) = ", len(t), len(z), len(mu), len(vpa))
    #print("gvmus.shape = ", gvmus.shape)

    #sys.exit()

    if ((which == "gvpa") or (which == "both")):
        ## Code to plot g for stella
        gvmus = gvmus[-1]   # spec, mu, vpa
        fig = plt.figure(figsize=[10,10])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        counter=0
        ion_max_val = 0
        electron_max_val = 0
        for mu_idx in range(0, len(mu)):
            counter += 1
            g_ion_vpa = gvmus[0, mu_idx, :]
            g_electron_vpa = gvmus[1, mu_idx, :]
            tmp_max_val = np.max(g_ion_vpa)
            ion_max_val = max(tmp_max_val, ion_max_val)
            tmp_max_val = np.max(g_electron_vpa)
            electron_max_val = max(tmp_max_val, electron_max_val)

            ax1.plot(vpa, g_ion_vpa, label="mu={:.3f}".format(mu[mu_idx]))
            if stretch_electron_vpa:
                ax2.plot(vpa/np.sqrt(0.00028), g_electron_vpa, label="mu={:.3f}".format(mu[mu_idx]))
            else:
                ax2.plot(vpa, g_electron_vpa, label="mu={:.3f}".format(mu[mu_idx]))

            if counter == 5:
                if plot_gauss_squared:
                    maxwell_vpa_squared = (np.exp(-vpa**2*0.00028))**2
                    ax1.plot(vpa, maxwell_vpa_squared*ion_max_val, c="black", ls="--", label=r"$\exp\left(-v_\parallel^2\right)$")
                    ax2.plot(vpa, maxwell_vpa_squared*electron_max_val, c="black", ls="--", label=r"$\exp\left(-v_\parallel^2\right)$")
                for ax in [ax1, ax2]:
                    ax.grid(True)
                    ax.legend(loc="best")
                ax2.set_xlabel("vpa")
                ax1.set_ylabel(r"$g_{ion}$")
                ax2.set_ylabel(r"$g_{electron}$")
                plt.show()
                fig = plt.figure(figsize=[10,10])
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                counter=0
                ion_max_val = 0
                electron_max_val = 0

        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.legend(loc="best")
        ax2.set_xlabel("vpa")
        ax1.set_ylabel(r"$g_{ion}$")
        ax2.set_ylabel(r"$g_{electron}$")
        plt.show()

    if ((which == "gmu") or (which == "both")):
        ###### Plot g(mu) for different vpa
        if which == "gmu":
            # Get the final timestep of gvmus
            gvmus = gvmus[-1]   # spec, mu, vpa
        fig = plt.figure(figsize=[10,10])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        counter=0
        #print("vpa = ", vpa)
        #sys.exit()
        # Only go over half of vpa-space (symmetry arguyment)
        for vpa_idx in range(int(len(vpa)/2), len(vpa)):
            counter += 1
            g_ion_mu = gvmus[0, :, vpa_idx]
            g_electron_mu = gvmus[1, :, vpa_idx]
            ax1.plot(mu, g_ion_mu, label="vpa={:.3f}".format(vpa[vpa_idx]))
            ax2.plot(mu, g_electron_mu, label="vpa={:.3f}".format(vpa[vpa_idx]))

            if counter == 5:
                for ax in [ax1, ax2]:
                    ax.grid(True)
                    ax.legend(loc="best")
                ax2.set_xlabel("mu")
                ax1.set_ylabel(r"$g_{ion}$")
                ax2.set_ylabel(r"$g_{electron}$")
                plt.show()
                fig = plt.figure(figsize=[10,10])
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                counter=0

        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.legend(loc="best")
        ax2.set_xlabel("mu")
        ax1.set_ylabel(r"$g_{ion}$")
        ax2.set_ylabel(r"$g_{electron}$")
        plt.show()

    return

def plot_gzvs(stella_outnc_longname, which="gvpa", plot_gauss_squared=False,
                stretch_electron_vpa=False):
    """ """
    view_ncdf_variables(stella_outnc_longname)
    t, z, vpa, gzvs = extract_data_from_ncdf(stella_outnc_longname,
                                    "t", 'zed', "vpa", 'gzvs')
    print("len(t)", "len(z), len(vpa) = ", len(t), len(z), len(vpa))
    print("gzvs.shape = ", gzvs.shape)  #
    sys.exit()

    if ((which == "gvpa") or (which == "both")):
        ## Code to plot g for stella
        gvmus = gvmus[-1]   # spec, mu, vpa
        fig = plt.figure(figsize=[10,10])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        counter=0
        ion_max_val = 0
        electron_max_val = 0
        for mu_idx in range(0, len(mu)):
            counter += 1
            g_ion_vpa = gvmus[0, mu_idx, :]
            g_electron_vpa = gvmus[1, mu_idx, :]
            tmp_max_val = np.max(g_ion_vpa)
            ion_max_val = max(tmp_max_val, ion_max_val)
            tmp_max_val = np.max(g_electron_vpa)
            electron_max_val = max(tmp_max_val, electron_max_val)

            ax1.plot(vpa, g_ion_vpa, label="mu={:.3f}".format(mu[mu_idx]))
            if stretch_electron_vpa:
                ax2.plot(vpa/np.sqrt(0.00028), g_electron_vpa, label="mu={:.3f}".format(mu[mu_idx]))
            else:
                ax2.plot(vpa, g_electron_vpa, label="mu={:.3f}".format(mu[mu_idx]))

            if counter == 5:
                if plot_gauss_squared:
                    maxwell_vpa_squared = (np.exp(-vpa**2*0.00028))**2
                    ax1.plot(vpa, maxwell_vpa_squared*ion_max_val, c="black", ls="--", label=r"$\exp\left(-v_\parallel^2\right)$")
                    ax2.plot(vpa, maxwell_vpa_squared*electron_max_val, c="black", ls="--", label=r"$\exp\left(-v_\parallel^2\right)$")
                for ax in [ax1, ax2]:
                    ax.grid(True)
                    ax.legend(loc="best")
                ax2.set_xlabel("vpa")
                ax1.set_ylabel(r"$g_{ion}$")
                ax2.set_ylabel(r"$g_{electron}$")
                plt.show()
                fig = plt.figure(figsize=[10,10])
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                counter=0
                ion_max_val = 0
                electron_max_val = 0

        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.legend(loc="best")
        ax2.set_xlabel("vpa")
        ax1.set_ylabel(r"$g_{ion}$")
        ax2.set_ylabel(r"$g_{electron}$")
        plt.show()

    if ((which == "gmu") or (which == "both")):
        ###### Plot g(mu) for different vpa
        if which == "gmu":
            # Get the final timestep of gvmus
            gvmus = gvmus[-1]   # spec, mu, vpa
        fig = plt.figure(figsize=[10,10])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        counter=0
        #print("vpa = ", vpa)
        #sys.exit()
        # Only go over half of vpa-space (symmetry arguyment)
        for vpa_idx in range(int(len(vpa)/2), len(vpa)):
            counter += 1
            g_ion_mu = gvmus[0, :, vpa_idx]
            g_electron_mu = gvmus[1, :, vpa_idx]
            ax1.plot(mu, g_ion_mu, label="vpa={:.3f}".format(vpa[vpa_idx]))
            ax2.plot(mu, g_electron_mu, label="vpa={:.3f}".format(vpa[vpa_idx]))

            if counter == 5:
                for ax in [ax1, ax2]:
                    ax.grid(True)
                    ax.legend(loc="best")
                ax2.set_xlabel("mu")
                ax1.set_ylabel(r"$g_{ion}$")
                ax2.set_ylabel(r"$g_{electron}$")
                plt.show()
                fig = plt.figure(figsize=[10,10])
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                counter=0

        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.legend(loc="best")
        ax2.set_xlabel("mu")
        ax1.set_ylabel(r"$g_{ion}$")
        ax2.set_ylabel(r"$g_{electron}$")
        plt.show()

    return
