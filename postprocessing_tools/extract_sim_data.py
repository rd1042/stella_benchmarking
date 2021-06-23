""" """

from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
import numpy as np
import sys

def get_omega_data(sim_longname, sim_type):
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
        #time, phi2 = extract_data_from_ncdf((sim_longname + ".out.nc"), "t", "phi2")
        time = omega_data[:,0]
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
    freq = omega.real
    gamma = omega.imag



    return time, freqom_final, gammaom_final, freq, gamma

def get_phiz_data(sim_longname, sim_type):
    """ """
    if sim_type == "stella":
        theta, real_phi, imag_phi = get_phiz_data_stella(sim_longname)
    elif sim_type == "gs2":
        theta, real_phi, imag_phi = get_phiz_data_gs2(sim_longname)
    else:
        print("sim_type not recognised!")

    return theta, real_phi, imag_phi

def get_aparz_data(sim_longname, sim_type):
    """ """
    if sim_type == "stella":
        theta, real_apar, imag_apar = get_aparz_data_stella(sim_longname)
    elif sim_type == "gs2":
        theta, real_apar, imag_apar = get_aparz_data_gs2(sim_longname)
    else:
        print("sim_type not recognised!")

    return theta, real_apar, imag_apar

def get_bparz_data(sim_longname, sim_type):
    """ """
    if sim_type == "stella":
        theta, real_bpar, imag_bpar = get_bparz_data_stella(sim_longname)
    elif sim_type == "gs2":
        theta, real_bpar, imag_bpar = get_bparz_data_gs2(sim_longname)
    else:
        print("sim_type not recognised!")

    return theta, real_bpar, imag_bpar

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

def find_bpar_phi_ratio(sim_longname, sim_type):
    """ """
    theta, real_phi, imag_phi = get_phiz_data(sim_longname, sim_type)
    theta, real_bpar, imag_bpar = get_bparz_data(sim_longname, sim_type)

    # get absolute values of the fields.

    # Perform initial normalisation so finite values do not become infinite
    # when squared
    initial_normalisation = np.max(real_phi)
    real_phi = real_phi/initial_normalisation
    imag_phi = imag_phi/initial_normalisation
    real_bpar = real_bpar/initial_normalisation
    imag_bpar = imag_bpar/initial_normalisation

    # Get the abs values
    abs_phi = np.sqrt(real_phi*real_phi + imag_phi*imag_phi)
    abs_bpar = np.sqrt(real_bpar*real_bpar + imag_bpar*imag_bpar)
    #print("")
    max_abs_phi = np.max(abs_phi)
    max_abs_bpar = np.max(abs_bpar)
    return max_abs_bpar/max_abs_phi

def find_apar_phi_ratio(sim_longname, sim_type):
    """ """
    theta, real_phi, imag_phi = get_phiz_data(sim_longname, sim_type)
    theta, real_apar, imag_apar = get_aparz_data(sim_longname, sim_type)

    # get absolute values of the fields.

    # Perform initial normalisation so finite values do not become infinite
    # when squared
    initial_normalisation = np.max(real_phi)
    real_phi = real_phi/initial_normalisation
    imag_phi = imag_phi/initial_normalisation
    real_apar = real_apar/initial_normalisation
    imag_apar = imag_apar/initial_normalisation

    # Get the abs values
    abs_phi = np.sqrt(real_phi*real_phi + imag_phi*imag_phi)
    abs_apar = np.sqrt(real_apar*real_apar + imag_apar*imag_apar)
    #print("")
    max_abs_phi = np.max(abs_phi)
    max_abs_apar = np.max(abs_apar)
    return max_abs_apar/max_abs_phi

def get_chi_stella():
    """ """
    return

def get_aparz_data_stella(sim_longname):
    """ """
    final_fields_filename = sim_longname + ".final_fields"
    final_fields_file = open(final_fields_filename, "r")
    final_fields_data=np.loadtxt(final_fields_filename,dtype='float')
    final_fields_file.close()

    ## final_fields_data = z, z-zed0, aky, akx, real(apar), imag(apar), real(apar), imag(apar),
    ## real(bpar), imag(bpar), z_eqarc-zed0, kperp2
    # Usually we're just looking at one mode; check how many unique kx and ky we have

    z = final_fields_data[:,0]

    aky = final_fields_data[:,2]; akx = final_fields_data[:,3]
    unique_aky = set(aky); unique_akx = set(akx)
    if len(unique_aky) > 1 or len(unique_akx) > 1:
        print("len(unique_aky), len(unique_akx) = ", len(unique_aky), len(unique_akx))
        print("Not currently supported")
        sys.exit()

    real_apar = final_fields_data[:,6]; imag_apar = final_fields_data[:,7]
    return z, real_apar, imag_apar

def get_aparz_data_gs2(sim_longname):
    """ """
    theta, apar = extract_data_from_ncdf((sim_longname + ".out.nc"), "theta","apar")
    if ((len(apar) > 1) or (len(apar[0]) > 1)):
        print("apar= ", apar)
        print("(len(apar) > 1) or (len(apar[0]) > 1), aborting")
    apar = apar[0,0,:]/2
    return theta, apar.real, apar.imag

def get_bparz_data_stella(sim_longname):
    """ """
    final_fields_filename = sim_longname + ".final_fields"
    final_fields_file = open(final_fields_filename, "r")
    final_fields_data=np.loadtxt(final_fields_filename,dtype='float')
    final_fields_file.close()

    ## final_fields_data = z, z-zed0, aky, akx, real(phi), imag(phi), real(apar), imag(apar),
    ## real(bpar), imag(bpar), z_eqarc-zed0, kperp2
    # Usually we're just looking at one mode; check how many unique kx and ky we have

    z = final_fields_data[:,0]

    aky = final_fields_data[:,2]; akx = final_fields_data[:,3]
    unique_aky = set(aky); unique_akx = set(akx)
    if len(unique_aky) > 1 or len(unique_akx) > 1:
        print("len(unique_aky), len(unique_akx) = ", len(unique_aky), len(unique_akx))
        print("Not currently supported")
        sys.exit()

    real_bpar = final_fields_data[:,8]; imag_bpar = final_fields_data[:,9]
    return z, real_bpar, imag_bpar

def get_bparz_data_gs2(sim_longname):
    """Get bpar(z) from the .out.nc file of a GS2 simulation,
    BUT normalised to be consistent with stella's bpar(z).
    ###################################################
    GS2 has bpar = B_parallel (Lref)/(B*rho_ref) = (Lref)/(bmag*Bref*rho_ref)
    stella has bpar = B_parallel (a)/(Bref*rho_ref)
    So stella's bpar is equivalent to GS2's bpar*bmag (assuming GS2's Lref, rho_ref
    is same as stella's!)
    """

    theta, bpar, bmag = extract_data_from_ncdf((sim_longname + ".out.nc"), "theta","bpar", "bmag")
    if ((len(bpar) > 1) or (len(bpar[0]) > 1)):
        print("bpar= ", bpar)
        print("(len(bpar) > 1) or (len(bpar[0]) > 1), aborting")
    bpar = bpar[0,0,:]*bmag
    return theta, bpar.real, bpar.imag
