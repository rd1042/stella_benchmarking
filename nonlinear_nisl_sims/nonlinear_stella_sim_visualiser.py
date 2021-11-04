""" """

import sys
sys.path.append("../postprocessing_tools")
from plotting_helper import make_comparison_plots, plot_gmvus, plot_gzvs
from plotting_helper import make_comparison_plots_leapfrog_poster
from helper_ncdf import view_ncdf_variables, extract_data_from_ncdf
from extract_sim_data import get_omega_data
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft

def make_phi2_kxky_modes_pics(outnc_longname):
    """ """
    # Get phi(kx, ky, z, t)
    [t, kx, ky, z, phi_vs_t] = extract_data_from_ncdf(outnc_longname, "t", 'kx', 'ky', "zed", "phi_vs_t")
    # print("phi_vs_t.shape = ", phi_vs_t.shape)  # time, tube (?), z, kx, ky
    # print("len(kx, ky, t, z) = ", len(kx), len(ky), len(t), len(z))
    nz_per_mode = len(z)
    z_idx = int(nz_per_mode/2)  # The z idx of z=0
    if z[z_idx] != 0:
        print("Error! z[z_idx] = ", z[z_idx])
        sys.exit()

    # z_idx = 1 # Old
    counter = 0
    phi_t_ky_kx = phi_vs_t[:, 0, z_idx, :, :]
    for ky_idx in range(0, len(ky)):
        for kx_idx in range(0, len(kx)):
            phi_t = phi_t_ky_kx[:, kx_idx, ky_idx]

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # ax2 = fig.add_subplot(312, sharex=ax1)
            # ax3 = fig.add_subplot(313, sharex=ax1)
            ax1.plot(t, phi_t.real, label="real(phi)")
            ax1.plot(t, phi_t.imag, label="im(phi)")
            ax1.plot(t, abs(phi_t), label="abs(phi)")

            ax1.set_xlabel(r"$t$")
            ax1.set_ylabel(r"$\phi$")
            ax1.legend(loc="best")
            fig.suptitle("kx={:.3f}, ky={:.3f}".format(kx[kx_idx], ky[ky_idx]))
            save_name="images/phi_t_{:02d}.png".format(counter)
            counter+=1
            plt.savefig(save_name)
            plt.close()

    return

def make_phi2_kxky_modes_pics_for_each_z(outnc_longname):
    """ """
    # Get phi(kx, ky, z, t)
    [t, kx, ky, z, phi_vs_t] = extract_data_from_ncdf(outnc_longname, "t", 'kx', 'ky', "zed", "phi_vs_t")
    # print("phi_vs_t.shape = ", phi_vs_t.shape)  # time, tube (?), z, kx, ky
    # print("len(kx, ky, t, z) = ", len(kx), len(ky), len(t), len(z))
    nz_per_mode = len(z)

    # z_idx = 1 # Old
    counter = 0
    for ky_idx in range(0, len(ky)):
        for kx_idx in range(0, len(kx)):

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # ax2 = fig.add_subplot(312, sharex=ax1)
            # ax3 = fig.add_subplot(313, sharex=ax1)
            for z_idx in range(0, nz_per_mode):
                ax1.plot(t, abs(phi_vs_t[:, 0, z_idx, kx_idx,ky_idx]))

            ax1.set_xlabel(r"$t$")
            ax1.set_ylabel(r"$\phi$")
            fig.suptitle("kx={:.3f}, ky={:.3f}".format(kx[kx_idx], ky[ky_idx]))
            save_name="images/phi_t_{:02d}.png".format(counter)
            counter+=1
            plt.savefig(save_name)
            plt.close()
            # plt.show()

    return

def examine_initialisation():
    """ """
    outnc_rk3_nz4 = "looking_at_initial_conditions/rk3_nonlinear_only_nz4.out.nc"
    outnc_nisl_nz4 = "looking_at_initial_conditions/nisl_nonlinear_only_nz4.out.nc"
    outnc_nisl_nz2 = "looking_at_initial_conditions/nisl_nonlinear_only_nz2.out.nc"

    # Get phi(kx, ky, z, t)
    [t_nisl, kx_nisl, ky_nisl, z_nisl, phi_vs_t_nisl] = extract_data_from_ncdf(outnc_nisl_nz4, "t", 'kx', 'ky', "zed", "phi_vs_t")
    [t_rk3, kx_rk3, ky_rk3, z_rk3, phi_vs_t_rk3] = extract_data_from_ncdf(outnc_rk3_nz4, "t", 'kx', 'ky', "zed", "phi_vs_t")
    [t_nz2, kx_nz2, ky_nz2, z_nz2, phi_vs_t_nz2] = extract_data_from_ncdf(outnc_nisl_nz2, "t", 'kx', 'ky', "zed", "phi_vs_t")
    # print("phi_vs_t.shape = ", phi_vs_t.shape)  # time, tube (?), z, kx, ky
    phi_nisl = phi_vs_t_nisl[0,0,:,:,:]
    phi_rk3 = phi_vs_t_rk3[0,0,:,:,:]
    # phi_diff = phi_nisl - phi_rk3
    # print("phi_diff = ", phi_diff)
    # print("kx_nisl[1], kx_nisl[-1] = ", kx_nisl[1], kx_nisl[-1])
    print("kx_nisl = ", kx_nisl)
    print("ky_nisl = ", ky_nisl)
    sys.exit()
    print("z, nzed=4 =  ", z_nisl)
    print("z, nzed=2 = ", z_nz2)
    print("phi_vs_t_nz2[0,0,:,1,1] = ", phi_vs_t_nz2[0,0,:,1,1])
    print("phi_vs_t_nz2[0,0,:,-1,1] = ", phi_vs_t_nz2[0,0,:,-1,1])
    # print("phi_nisl[:,1,1] = ", phi_nisl[:,1,1])
    # print("phi_nisl[:,-1,1] = ", phi_nisl[:,-1,1])
    # print("len(kx, ky, t, z) = ", len(kx), len(ky), len(t), len(z))

def ifft_phi(kx, ky, phi_kxky):
    """ """
    # Before FTing, we need to increase the size of phi_kxky; this is because
    # ifft2 is expecting both kx and ky to include negative values i.e.
    # expects kx = (0, dkx, 2*dkx, . . ., kxmax, -kxmax, -kxmax+dkx, . . . , -dkx)
    # and     ky = (0, dky, 2*dky, . . ., kymax, -kymax, -kymax+dky, . . . , -dky)
    # whereas currently ky has only positive values. This is becasue g(x,y) is
    # purely real, and so the entries for the -ve values of ky are just the
    # complex conjugate of the +ve values, so can be ignored.
    # Question: How to do this correctly to ensure g(x,y) purely real? Have
    # tried padding phi_kxky with the complex conjugate of phi_kxky, but
    # this seems not to work. stella performs the transform in a rather curious
    # way, so could look at stella's routine and adopt it for ourselves?
    # stella seems to:
    # (1) "swap" kx,ky. This consists of changing the shape from (naky, nakx)
    # to (2*naky-1, ikx_max); so going from
    #    kx=(0, . . ., kxmax, -kxmax, . . . , -dkx) , ky=(0, . . ., kymax)
    # to
    #    kx=(0, . . ., kxmax) , ky=(0, . . ., kymax, . . . , -dky)
    # The -ve ky values for a particular kx look to be the complex conjugate
    # of the -ve kx vals for that kx (with kx=0 being treated specially.)
    # (2) Perform a complex-to-complex transformation in y. This is straightforward,
    # except for the padding with zeros to avoid de-aliasing.
    # (3) Perform a complex-to-real transformation in x. It also looks like this
    # is just padding with zeros to avoid de-aliasing; the size of the transformed
    # array is nx/2+1 and the number of non-zero entries is (nakx/2+1). HOWEVER,
    # the output of the tranformation is size nx; so this DOESN'T look like an
    # ordinary FT which just throws away the complex part.

    # First, do the swap. Copied from swap_kxky_complex in ktgrids.f90
    nkx_inc_negative = len(kx) ; nky_no_negative = len(ky)
    nkx_no_negative = int((nkx_inc_negative + 1)/2)
    nky_inc_negative = int(nky_no_negative*2 - 1)
    # print("nkx_no_negative, nkx_inc_negative, nky_no_negative, nky_inc_negative = ",
    #         nkx_no_negative, nkx_inc_negative, nky_no_negative, nky_inc_negative)
    # NB phi_kxky is shape(nkx_inc_negative, nky_no_negative)
    phi_kxky_swap = np.zeros((nkx_no_negative, nky_inc_negative), dtype="complex")
    # +ve kx, ky entries are the same
    phi_kxky_swap[:nkx_no_negative, :nky_no_negative] = phi_kxky[:nkx_no_negative, :nky_no_negative]
    # Treat kx=0 specially
    for iky in range(nky_no_negative,nky_inc_negative):
        # Want the ky idx corresponding to the +ve ky value of ky(iky)
        # e.g. for iky=nky_no_negative, ky=-kymax, so want the ky idx corresponding
        # to +kymax (in this example, ikyneg=nky_no_negative-1)
        ikyneg = nky_inc_negative - iky
        #print("iky, ikyneg = ", iky, ikyneg)
        phi_kxky_swap[0,iky] = np.conj(phi_kxky[0,ikyneg])

    # Now map the (-ve kx, +ve ky) values to (+ve kx, -ve ky) values
    for ikx in range(1, nkx_no_negative):
        # Get the kx idx corresponding to the -ve kx value of kx(ikx)
        # e.g. for ikx=nkx_no_negative-1 , kx=kxmax so want the kx idx
        # corresponding to -kxmax (in this case, nkx_no_negative)
        ikxneg = nkx_inc_negative - ikx
        for iky in range(nky_no_negative,nky_inc_negative):
            ikyneg = nky_inc_negative - iky
            phi_kxky_swap[ikx,iky] = np.conj(phi_kxky[ikxneg,ikyneg])

    # Now FT in y
    phi_kxy = np.zeros((nkx_no_negative, nky_inc_negative), dtype="complex")
    for ikx in range(0, nkx_no_negative):
        phi_kxy[ikx,:] = fft.ifft(phi_kxky_swap[ikx,:])

    # Now FT in x. Careful, because we want "c2r"; I think we can achieve this
    # by padding with the complex conjugate.
    phi_xy = np.zeros((nkx_inc_negative, nky_inc_negative), dtype="complex")
    for iky in range(0, nky_inc_negative):
        phi_kxy_padded = np.zeros((nkx_inc_negative), dtype="complex")
        phi_kxy_padded[:nkx_no_negative] = phi_kxy[:,iky]
        conjugate_vals = np.conj((phi_kxy[1:,iky])[::-1])
        #print("phi_kxy_padded, conjugate_vals = ", phi_kxy_padded, conjugate_vals)
        phi_kxy_padded[nkx_no_negative:] = conjugate_vals
        phi_xy[:,iky] = fft.ifft(phi_kxy_padded)


    # phi_kxky_full = np.zeros((nx, nky_full),dtype="complex")
    # # print("phi_kxky_full.shape = ", phi_kxky_full.shape)
    # phi_kxky_full[:,:nky] = phi_kxky
    # higher_entries = phi_kxky[:,1:]
    # # Reverse the order
    # print("higher_entries[0,:] = ", higher_entries[0,:])
    # higher_entries = higher_entries[:,::-1]
    # print("higher_entries[0,:] = ", higher_entries[0,:])
    # phi_kxky_full[:,nky:] = np.conj(higher_entries)

    #phi_xy = (fft.ifft2(phi_kxky_full))
    if np.max(abs(phi_xy.imag) > 1e-10):
        print("error! Complex part too big")
        print("phi_xy = ", phi_xy)
        sys.exit()
    # else:
    #     print("Success!")
    #     #sys.exit()
    return phi_xy.real

def make_phi2_movie(outnc_longname):
    """ """
    # Get phi(kx, ky, z, t)
    [t, kx, ky, z, phi_vs_t] = extract_data_from_ncdf(outnc_longname, "t", 'kx', 'ky', "zed", "phi_vs_t")
    # print("phi_vs_t.shape = ", phi_vs_t.shape)  # time, tube (?), z, kx, ky
    # print("len(kx, ky, t, z) = ", len(kx), len(ky), len(t), len(z))

    z_idx = 1
    counter = 0
    phi_t_ky_kx = phi_vs_t[:, 0, z_idx, :, :]
    # print("phi_t_ky_kx[0,:,:] = ", phi_t_ky_kx[0,:,:])
    # sys.exit()
    # For each time, perform the inverse Fourier transform to get phi(x,y). Make
    # this into a colorplot, and save it.
    print("kx = ", kx)
    print("ky = ", ky)
    dx = np.pi/np.max(kx)
    dy = np.pi/np.max(ky)
    nx = len(kx)
    ny = len(ky); nky=ny
    nky_full = nky*2-1  # length of ky including -ve freqs
    ny_full = nky_full
    xmax = (nx-1)*dx
    ymax = (ny_full-1)*dy
    xvals = np.linspace(0, xmax, nx)
    yvals = np.linspace(0, ymax, ny_full)
    # print("xvals = ", xvals)
    # print("yvals = ", yvals)
    xmesh, ymesh = np.meshgrid(xvals,yvals)

    # Want the colorbar to be the same for all time, and centered about zero.
    phi_xy_t = np.zeros((ny_full,nx,len(t)))

    for t_idx in range(0, len(t), 1):
        phi_kxky = phi_t_ky_kx[t_idx, :,:]
        phi_xy = ifft_phi(kx, ky, phi_kxky)
        # print("phi_xy = ", phi_xy)
        # sys.exit()
        # Need to swap rows and columns; each column should be a single x
        phi_xy = phi_xy.T
        phi_xy_t[:,:,t_idx] = phi_xy

    max_phi = np.max(abs(phi_xy_t))
    print("max_phi = ", max_phi)

    for t_idx in range(0, len(t), 1):

        # x = fft.ifft(kx)
        # y = fft.ifft(ky)
        # print("x = ", x)
        # print("y = ", y)
        #print("phi_xy.shape = ", phi_xy.shape)
        #print("phi_xy = ", phi_xy)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax2 = fig.add_subplot(312, sharex=ax1)
        # ax3 = fig.add_subplot(313, sharex=ax1)
        pcm = ax1.pcolormesh(xvals, yvals, phi_xy_t[:,:,t_idx], cmap='RdBu_r', vmin=-max_phi, vmax=max_phi)
        fig.colorbar(pcm, ax=ax1)
        #ax1.contourf(xmesh, ymesh, phi_xy)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        fig.suptitle("t={:.3f}".format(t[t_idx]))
        save_name="images/phi_t_{:02d}.png".format(counter)
        counter+=1
        #plt.show()
        # sys.exit()
        plt.savefig(save_name)
        plt.close()

    return

def compare_sims_with_different_nwrite():
    """ """
    nwrite100 = "example_nisl_nonlinear_only_vexb10_for_visualisation.out.nc"
    nwrite51 = "example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite51.out.nc"
    nwrite25 = "example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite25.out.nc"
    nwrite1 = "example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite1.out.nc"
    nwrite1_smallerdt = "example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite1_delt0.015.out.nc"

    [t_nwrite100, kx_nwrite100, ky_nwrite100, z_nwrite100, phi_vs_t_nwrite100] = extract_data_from_ncdf(nwrite100, "t", 'kx', 'ky', "zed", "phi_vs_t")
    [t_nwrite51, kx_nwrite51, ky_nwrite51, z_nwrite51, phi_vs_t_nwrite51] = extract_data_from_ncdf(nwrite51, "t", 'kx', 'ky', "zed", "phi_vs_t")
    [t_nwrite25, kx_nwrite25, ky_nwrite25, z_nwrite25, phi_vs_t_nwrite25] = extract_data_from_ncdf(nwrite25, "t", 'kx', 'ky', "zed", "phi_vs_t")
    [t_nwrite1, kx_nwrite1, ky_nwrite1, z_nwrite1, phi_vs_t_nwrite1] = extract_data_from_ncdf(nwrite1, "t", 'kx', 'ky', "zed", "phi_vs_t")
    [t_nwrite1_smallerdt, kx_nwrite1_smallerdt, ky_nwrite1_smallerdt,
        z_nwrite1_smallerdt, phi_vs_t_nwrite1_smallerdt] = extract_data_from_ncdf(nwrite1_smallerdt, "t", 'kx', 'ky', "zed", "phi_vs_t")

    # Plot |phi(t)| for each z, kx, ky
    nz_per_mode = len(z_nwrite51)

    z_idxs = [int(nz_per_mode/2)]  # The z idx of z=0
    if z_nwrite51[z_idxs[0]] != 0:
        print("Error! z[z_idx] = ", z[z_idx])
        sys.exit()
    # z_idx = 1 # Old
    counter = 0
    for ky_idx in range(0, len(ky_nwrite51)):
        for kx_idx in range(0, len(kx_nwrite51)):

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # ax2 = fig.add_subplot(312, sharex=ax1)
            # ax3 = fig.add_subplot(313, sharex=ax1)
            #for z_idx in range(0, nz_per_mode):
            ## For now, only plot for z=0

            for z_idx in z_idxs:
                # ax1.plot(t_nwrite100, abs(phi_vs_t_nwrite100[:, 0, z_idx, kx_idx,ky_idx]))
                # ax1.plot(t_nwrite51, abs(phi_vs_t_nwrite51[:, 0, z_idx, kx_idx,ky_idx]), ls="--")
                # ax1.plot(t_nwrite25, abs(phi_vs_t_nwrite25[:, 0, z_idx, kx_idx,ky_idx]), ls="-.")
                ax1.plot(t_nwrite1, abs(phi_vs_t_nwrite1[:, 0, z_idx, kx_idx,ky_idx]), ls="-", c="black", label="dt=0.03")
                ax1.plot(t_nwrite1_smallerdt, abs(phi_vs_t_nwrite1_smallerdt[:, 0, z_idx, kx_idx,ky_idx]), ls="-", c="red", label="dt=0.015")
                # ax1.plot(t_nwrite1, (phi_vs_t_nwrite1[:, 0, z_idx, kx_idx,ky_idx]).real, label="real(phi)")
                # ax1.plot(t_nwrite1, (phi_vs_t_nwrite1[:, 0, z_idx, kx_idx,ky_idx]).imag, label="im(phi)")
            ax1.set_xlabel(r"$t$")
            ax1.set_ylabel(r"$\phi$")
            ax1.legend(loc="best")
            fig.suptitle("kx={:.3f}, ky={:.3f}".format(kx_nwrite51[kx_idx], ky_nwrite51[ky_idx]))
            save_name="images/phi_t_{:02d}.png".format(counter)
            counter+=1
            #plt.savefig(save_name)
            #plt.close()
            plt.show()

    return



if __name__ == "__main__":
    print("Hello world")
    #make_phi2_kxky_modes_pics("example_rk3_nonlinear_only_vexb1_for_visualisation.out.nc")
    #make_phi2_kxky_modes_pics("example_rk3_nonlinear_only_vexb10_for_visualisation_longer_time.out.nc")
    examine_initialisation()
    #make_phi2_kxky_modes_pics("example_nisl_nonlinear_only_vexb1_for_visualisation.out.nc")
    #make_phi2_kxky_modes_pics("example_nisl_nonlinear_only_vexb10_for_visualisation.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("example_nisl_nonlinear_only_vexb10_for_visualisation.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite51.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("example_nisl_nonlinear_only_vexb10_for_visualisation_nwrite1.out.nc")
    #compare_sims_with_different_nwrite()
    # make_phi2_kxky_modes_pics_for_each_z("example_rk3_nonlinear_only_vexb10_single_mode.out.nc")
    # make_phi2_kxky_modes_pics_for_each_z("example_nisl_nonlinear_only_vexb10_single_mode.out.nc")
    # make_phi2_kxky_modes_pics_for_each_z("example_nisl_nonlinear_only_vexb10_single_mode_kxky0.667.out.nc")
    #make_phi2_kxky_modes_pics_for_each_z("example_nisl_nonlinear_only_vexb1_for_visualisation.out.nc")
    #make_phi2_movie("example_rk3_nonlinear_only_vexb1_for_visualisation.out.nc")
    #make_phi2_movie("example_rk3_nonlinear_only_vexb10_for_visualisation_longer_time.out.nc")
    #make_phi2_movie("example_rk3_nonlinear_only_vexb_x0_y10_single_mode.out.nc")
