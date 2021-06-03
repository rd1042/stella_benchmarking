""" """

from extract_sim_data import get_omega_data, get_phiz_data
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import sys

## Define linestyles
linestyles1=cycle(["-", "--", "-.", ":"])
linestyles2=cycle(["-", "--", "-.", ":"])




def plot_omega_t_for_sim(ax1, ax2, sim_longname, sim_label, sim_type="stella"):
    """ """
    time, freqom_final, gammaom_final, freqom, gammaom = get_omega_data(sim_longname, sim_type=sim_type)
    print("freqom_final, gammaom_final = ", freqom_final, gammaom_final)
    half_len = int(len(time)/2)
    freqom_half = freqom[half_len:]
    gammaom_half = gammaom[half_len:]
    # Get an estimate for the ylims by looking at the max/min of the second half
    # of the frequencuy and gamma data
    linestyle=next(linestyles1)
    gamma_llim = (np.min(gammaom_half)); gamma_ulim = (np.max(gammaom_half))
    freq_llim = (np.min(freqom_half)); freq_ulim = np.max(freqom_half)
    ax1.plot(time, freqom, label=sim_label, ls=linestyle)
    ax2.plot(time, gammaom, label=sim_label, ls=linestyle)
    return gammaom_final, freqom_final, gamma_llim, gamma_ulim, freq_llim, freq_ulim

def plot_phi_z_for_sim(ax1, sim_longname, sim_label, sim_type="stella"):
    """ """

    theta, real_phi, imag_phi = get_phiz_data(sim_longname, sim_type=sim_type)
    ## Check values are finite
    if not(np.all(np.isfinite(real_phi)) and np.all(np.isfinite(imag_phi))):
        print("Error! phi contains non-finite values")
        sys.exit()

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



def make_comparison_plots(sim_longnames, sim_labels, save_name, sim_types=[]):
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
        gamma_llims.append(gamma_llim)
        gamma_ulims.append(gamma_ulim)
        freq_llims.append(freq_llim)
        freq_ulims.append(freq_ulim)
        gamma_vals.append(gammaom_final)
        freq_vals.append(freqom_final)

    ## Set lims based on sim data
    gamma_llim = np.min(np.array(gamma_llims))
    gamma_ulim = np.max(np.array(gamma_ulims))
    freq_llim = np.min(np.array(freq_llims))
    freq_ulim = np.max(np.array(freq_ulims))
    ax11.set_ylim(freq_llim, freq_ulim)
    ax12.set_ylim(gamma_llim, gamma_ulim)
    ax21.set_ylim(-0.05, 1.05)
    ax12.set_xlabel(r"$t$")
    ax11.set_ylabel(r"$\omega$")
    ax12.set_ylabel(r"$\gamma$")
    ax21.set_xlabel(r"$\theta/\pi$")
    ax21.set_ylabel(r"$\vert \phi \vert$")
    for ax in [ax11, ax12, ax21]:
        ax.grid(True)
        ax.legend(loc="best")
    fig1.savefig(save_name + "_omega.eps")
    fig2.savefig(save_name + "_phi2.eps")
    plt.close(fig1)
    plt.close(fig2)
    return

def make_beta_scan_plots(sim_longnames, beta_vals, save_name, sim_types=[]):
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
        gamma_llims.append(gamma_llim)
        gamma_ulims.append(gamma_ulim)
        freq_llims.append(freq_llim)
        freq_ulims.append(freq_ulim)
        gamma_vals.append(gammaom_final)
        freq_vals.append(freqom_final)

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

    ax21.plot(beta_vals, freq_vals)
    ax22.plot(beta_vals, gamma_vals)
    ax21.set_ylabel(r"$\omega$")
    ax22.set_ylabel(r"$\gamma$")
    ax22.set_xlabel(r"$\beta$")

    for ax in [ax11, ax12, ax21, ax22]:
        ax.grid(True)
    for ax in [ax11, ax12]:
        ax.legend(loc="best")

    fig1.savefig(save_name + "_omega_t.eps")
    fig2.savefig(save_name + "_omega_beta.eps")
    plt.close(fig1)
    plt.close(fig2)

    return
