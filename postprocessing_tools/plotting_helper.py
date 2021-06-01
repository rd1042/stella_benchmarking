""" """

from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import numpy as np
import matplotlib.pyplot as plt

def process_omega_file(long_sim_name):
    """The .omega file contains columns t, ky, kx, re(omega), im(omega), av re(omega), av im(omega).
    (NB. Some .omega files contain NaN values)
    Read this file, and put data in arrays. We also want to calculate
    the growth rate from the phi2_vs_kxky entry in the .out.nc file."""
    omega_filename = long_sim_name + ".omega"
    omega_file = open(omega_filename, "r")
    data=np.loadtxt(omega_filename,dtype='float')
    omega_file.close()

    # Find the number of unique kx, kx, and construct omega-related arrays with dims (kx.ky)
    kx_array = np.array(sorted(set(data[:,2])))
    ky_array = np.array(sorted(set(data[:,1])))
    if ((len(kx_array) * len(ky_array)) > 1):
        freqom_kxky = np.zeros((len(kx_array), len(ky_array)))
        gammaom_kxky = np.zeros((len(kx_array), len(ky_array)))
        gamma_p2_kxky = np.zeros((len(kx_array), len(ky_array)))

        # Extract t and phi2(t) from .out.nc
        outnc_t, phi2_kxky = extract_data_from_ncdf((long_sim_name + ".out.nc"), "t", "phi2_kxky")

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
                finite_mode_t = outnc_t[finite_mode_phi2_idxs]

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
        outnc_t, phi2 = extract_data_from_ncdf((long_sim_name + ".out.nc"), "t", "phi2")
        freqom_final = data[-1, 3]
        gammaom_final = data[-1, 4]

        return outnc_t, freqom_final, gammaom_final, data[:,3], data[:,4]

def plot_omega_t_for_sim(sim_longname, ax1, ax2):
    """ """
    outnc_t, freqom_final, gammaom_final, freqom, gammaom = process_omega_file(sim_longname)
    print("freqom_final, gammaom_final = ", freqom_final, gammaom_final)
    half_len = int(len(outnc_t)/2)
    freqom_half = freqom[half_len:]
    gammaom_half = gammaom[half_len:]
    # Get an estimate for the ylims by looking at the max/min of the second half
    # of the frequencuy and gamma data
    gamma_llim = (np.min(gammaom_half)); gamma_ulim = (np.max(gammaom_half))
    freq_llim = (np.min(freqom_half)); freq_ulim = np.max(freqom_half)
    ax1.plot(outnc_t[1:], freqom[:])
    ax2.plot(outnc_t[1:], gammaom[:])
    return gammaom_final, freqom_final, gamma_llim, gamma_ulim, freq_llim, freq_ulim

def make_comparison_plots(sim_longnames, labels):
    """Compare multiple simulations which have a single common input. Create the following
    plots:
    1) omega(t)
    2) Normalised phi2(z)
    """

    ## Plot of omega(t)
    fig1 = plt.figure()
    ax11 = fig1.add_subplot(211)
    ax12 = fig1.add_subplot(212, sharex=ax11)

    ## Plot of phi2(t)
    fig2 = plt.figure()
    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212, sharex=ax21)
    gamma_vals = []
    freq_vals = []

    gamma_llims = []
    gamma_ulims = []
    freq_llims = []
    freq_ulims = []

    for sim_longname in sim_longnames:
        gammaom_final, freqom_final, gamma_llim, gamma_ulim, freq_llim, freq_ulim = plot_omega_t_for_sim(sim_longname, ax11, ax12)
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
    for ax in [ax11, ax12, ax21, ax22]:
        ax.grid(True)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    fig1.savefig(save_name + "_omega.eps")
    fig2.savefig(save_name + "_phi2.eps")
    fig1.close()
    fig2.close()
    return
