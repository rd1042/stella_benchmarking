""" """

from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import sys

## Define linestyles
linestyles1=cycle(["-", "--", "-.", ":"])
linestyles2=cycle(["-", "--", "-.", ":"])


def get_omega_data(sim_longname, sim_type="stella"):
    """ """

    if sim_type == "stella":
        time, freqom_final, gammaom_final, freqom, gammaom = get_omega_data_stella(sim_longname)
    elif sim_type == "gs2":
        time, freqom_final, gammaom_final, freqom, gammaom = get_omega_data_gs2(sim_longname)
    else:
        print("sim_type not recognised!")

    return time, freqom_final, gammaom_final, freqom, gammaom

def get_omega_data_stella(sim_longname):
    """The .omega file contains columns t, ky, kx, re(omega), im(omega), av re(omega), av im(omega).
    (NB. Some .omega files contain NaN values)
    Read this file, and put data in arrays. We also want to calculate
    the growth rate from the phi2_vs_kxky entry in the .out.nc file."""
    omega_filename = sim_longname + ".omega"
    omega_file = open(omega_filename, "r")
    omega_data=np.loadtxt(omega_filename,dtype='float')
    omega_file.close()

    # Find the number of unique kx, kx, and construct omega-related arrays with dims (kx.ky)
    kx_array = np.array(sorted(set(omega_data[:,2])))
    ky_array = np.array(sorted(set(omega_data[:,1])))
    if ((len(kx_array) * len(ky_array)) > 1):
        print("Multiple modes - not currently supproted!")
        sys.exit()
        freqom_kxky = np.zeros((len(kx_array), len(ky_array)))
        gammaom_kxky = np.zeros((len(kx_array), len(ky_array)))
        gamma_p2_kxky = np.zeros((len(kx_array), len(ky_array)))

        # Extract t and phi2(t) from .out.nc
        time, phi2_kxky = extract_data_from_ncdf((sim_longname + ".out.nc"), "t", "phi2_kxky")

        # Loop through the lists, putting values into the arrays
        # NB we assume that stella changes t, then ky, then kx when writing to file
        print("(len(kx_array) * len(ky_array)) = ", (len(kx_array) * len(ky_array)))
        for kx_idx in range(0, len(kx_array)):
            for ky_idx in range(0, len(ky_array)):

                # Find real(omega) and im(omega) for this kx, ky, ignoring nans and infs
                # Also find the finite phi2 entries, and their corresponding t values.
                mode_data = data[np.where((data[:,2] == kx_array[kx_idx]) & (data[:,1] == ky_array[ky_idx]))]
                finite_mode_data = mode_data[np.where((np.isfinite(mode_data[:,3])) & (np.isfinite(mode_data[:,4])))]
                mode_phi2 = phi2_kxky[:,kx_idx,ky_idx]
                finite_mode_phi2_idxs = np.where(np.isfinite(mode_phi2))
                fintite_mode_phi2 = mode_phi2[finite_mode_phi2_idxs]
                finite_mode_t = time[finite_mode_phi2_idxs]

                # If phi2 is too small, stella porbably has trouble fitting
                # an expontential to it. Filter out low-phi2 values to avoid noise.
                if fintite_mode_phi2[-1] < min_phi2:
                    freqom_kxky[kx_idx, ky_idx] = 0
                    gammaom_kxky[kx_idx, ky_idx] = 0
                else:
                    freqom_kxky[kx_idx, ky_idx] = finite_mode_data[-1,3]
                    gammaom_kxky[kx_idx, ky_idx] = finite_mode_data[-1,4]

                # Calculate growth rate based on phi2(t).
                gamma_p2_kxky[kx_idx, ky_idx] = (1/(2*(finite_mode_t[-1]-finite_mode_t[0])) *
                                    (np.log(fintite_mode_phi2[-1]) - np.log(fintite_mode_phi2[0])) )
        return [kx_array, ky_array, freqom_kxky, gammaom_kxky, gamma_p2_kxky]
    else:
        time, phi2 = extract_data_from_ncdf((sim_longname + ".out.nc"), "t", "phi2")
        freqom_final = omega_data[-1, 3]
        gammaom_final = omega_data[-1, 4]

        return time, freqom_final, gammaom_final, omega_data[:,3], omega_data[:,4]

def get_omega_data_gs2(sim_longname):
    """ """

    try:
        time, omega = extract_data_from_ncdf((sim_longname + ".out.nc"), "t", "omega_average")
    except KeyError:
        time, omega = extract_data_from_ncdf((sim_longname + ".out.nc"), "t", "omegaavg")

    ## Check that there's only 1 mode i.e. omega has dimensions (n_time, 1, 1)
    if ((len(omega[0]) > 1) or (len(omega[0][0]) > 1)):
        print("omega[0]= ", omega[0])
        print("len(omega[0]) > 1 or len(omega[0][0]) > 1, aborting")
        sys.exit()
    omega = omega[:,0,0]
    freqom_final = omega[-1].real
    gammaom_final = omega[-1].imag
    freq = omega[1:].real
    gamma = omega[1:].imag



    return time, freqom_final, gammaom_final, freq, gamma

def get_phiz_data(sim_longname, sim_type="stella"):
    """ """
    if sim_type == "stella":
        theta, real_phi, imag_phi = get_phiz_data_stella(sim_longname)
    elif sim_type == "gs2":
        theta, real_phi, imag_phi = get_phiz_data_gs2(sim_longname)
    else:
        print("sim_type not recognised!")

    return theta, real_phi, imag_phi

def get_phiz_data_stella(sim_longname):
    """ """
    final_fields_filename = sim_longname + ".final_fields"
    final_fields_file = open(final_fields_filename, "r")
    final_fields_data=np.loadtxt(final_fields_filename,dtype='float')
    final_fields_file.close()

    ## final_fields_data = z, z-zed0, aky, akx, real(phi), imag(phi), real(apar), imag(apar), z_eqarc-zed0, kperp2
    # Usually we're just looking at one mode; check how many unique kx and ky we have

    z = final_fields_data[:,0]

    aky = final_fields_data[:,2]; akx = final_fields_data[:,3]
    unique_aky = set(aky); unique_akx = set(akx)
    if len(unique_aky) > 1 or len(unique_akx) > 1:
        print("len(unique_aky), len(unique_akx) = ", len(unique_aky), len(unique_akx))
        print("Not currently supported")
        sys.exit()

    real_phi = final_fields_data[:,4]; imag_phi = final_fields_data[:,5]
    return z, real_phi, imag_phi

def get_phiz_data_gs2(sim_longname):
    """ """
    theta, phi = extract_data_from_ncdf((sim_longname + ".out.nc"), "theta","phi")
    if ((len(phi) > 1) or (len(phi[0]) > 1)):
        print("phi= ", phi)
        print("(len(phi) > 1) or (len(phi[0]) > 1), aborting")
    phi = phi[0,0,:]
    return theta, phi.real, phi.imag

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
    ax1.plot(time[1:], freqom[:], label=sim_label, ls=linestyle)
    ax2.plot(time[1:], gammaom[:], label=sim_label, ls=linestyle)
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
    print("In make_comparison_plots")
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

    print("Plotted omega")
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
