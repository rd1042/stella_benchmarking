"""Compare 3 sets of data:
(1) golder
(2) gnew, SL only
(3) gnew, SL + RHS
 """
import re
import matplotlib.pyplot as plt
import numpy as np

ny = 14
nx = 10


dx =    1.8844221105527634 ; dy = 1.3463968515384828
y = np.array((0.0000000000000000, 1.3463968515384828, 2.6927937030769655, 4.0391905546154483,
             5.3855874061539311, 6.7319842576924138, 8.0783811092308966, 9.4247779607693793,
             10.771174812307862, 12.117571663846345, 13.463968515384828, 14.810365366923310,
             16.156762218461793, 17.503159070000276))
x = np.array((0.0000000000000000, 1.8844221105527634, 3.7688442211055269, 5.6532663316582905,
              7.5376884422110537, 9.4221105527638169, 11.306532663316581, 13.190954773869343,
              15.075376884422107, 16.959798994974872))

code_dt = 0.05

x_2d = np.zeros((ny, nx))
y_2d = np.zeros((ny, nx))
x_idxs = np.arange(0, nx, 1, dtype="int")
y_idxs = np.arange(0, ny, 1, dtype="int")
x_idxs_2d = np.zeros((ny,nx), dtype="int")
y_idxs_2d = np.zeros((ny,nx), dtype="int")

for ix in range(0, nx):
    x_2d[:,ix] = x[ix]
    y_idxs_2d[:,ix] = y_idxs
for iy in range(0, ny):
    x_idxs_2d[iy, :] = x_idxs
    y_2d[iy,:] = y[iy]

def plot_arrays_line_by_line_from_files(golder_array, gnew_slonly_array_stella,
                                    gnew_array_stella, gnew_array_tracking_traj_stella):
    """ """

    gdiff1 = gnew_array_stella - golder_array
    #gdiff2 = gnew_slonly_array_stella - golder_array
    gdiff3 = gnew_array_tracking_traj_stella - golder_array
    gdiff4 = gnew_array_tracking_traj_stella - gnew_array_stella
    print("gdiff1 = ", gdiff1)
    print("gdiff3 = ", gdiff3)
    print("gdiff4 = ", gdiff4)

    for yidx in range(0, ny):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.plot(range(0,nx), golder_array[yidx], label="golder", lw=4)
        ax1.plot(range(0,nx), gnew_slonly_array_stella[yidx], label="gnew_sl_only_array", lw=4)
        ax1.plot(range(0,nx), gnew_array_tracking_traj_stella[yidx], label="gnew_array", ls="--", lw=4)
        ax2.plot(range(0,nx), gdiff1[yidx,:], label="gnew - golder", lw=4)
        # ax2.plot(range(0,nx), gdiff2[yidx,:], label="gnew_sl_only - golder", lw=4)

        for ax in [ax1, ax2]:
            ax.legend(loc="best")
            ax.grid(True)
        plt.show()

    return

def plot_arrays_line_by_line_compare_to_python(golder_array, gnew_slonly_array_stella, gnew_array_stella,
                                           gnew_slonly_array_python, gnewyx_array_python):
    """ """

    gdiff1 = gnew_array_stella - golder_array
    gdiff2 = gnew_array_stella - gnewyx_array_python
    gdiff3 = gnew_slonly_array_stella - gnew_slonly_array_python
    gdiff4 = gnew_array_python - golder_array
    # print("gnew_array_stella - golder_array = ", gdiff1)
    print("gnew_array_stella - gnewyx_array_python = ", gdiff2)
    # print("gnew_slonly_array_stella - gnew_slonly_array_python = ", gdiff3)
    print("gnew_array_python - golder_array = ", gdiff4)

    # for yidx in range(0, ny):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(211)
    #     ax2 = fig.add_subplot(212, sharex=ax1)
    #     ax1.plot(range(0,nx), golder_array[yidx], label="golder", lw=4)
    #     ax1.plot(range(0,nx), gnew_sl_only_array[yidx], label="gnew_sl_only_array", lw=4)
    #     ax1.plot(range(0,nx), gnew_array[yidx], label="gnew_array", ls="--", lw=4)
    #     ax2.plot(range(0,nx), gdiff1[yidx,:], label="gnew - golder", lw=4)
    #     # ax2.plot(range(0,nx), gdiff2[yidx,:], label="gnew_sl_only - golder", lw=4)
    #
    #     for ax in [ax1, ax2]:
    #         ax.legend(loc="best")
    #         ax.grid(True)
    #     plt.show()

    return

def visualise_nisl_calculation(golder_array, dgdxold_array, dgdyold_array,
                                vx_array, vy_array):
    """Given the array dg/dx, dg/dy, v_x, v_y, golder, advance NISL nonlinearly"""
    print("Hello world")

    v_scaling_fac = 0.7
    vx_array = vx_array*v_scaling_fac
    vy_array = vy_array*v_scaling_fac

    x_departure = x_2d - vx_array*2*code_dt
    y_departure = y_2d - vy_array*2*code_dt
    # p_array = np.zeros((ny, nx), dtype="int")
    # q_array = np.zeros((ny, nx), dtype="int")
    p_array = np.rint((x_2d - x_departure)/dx).astype("int")
    q_array = np.rint((y_2d - y_departure)/dy).astype("int")
    xidx_for_norm_array = (x_idxs_2d - p_array)%nx
    yidx_for_norm_array = (y_idxs_2d - q_array)%ny
    # xidx_for_upsampled_array = (2*x_idxs_2d)%(2*nx)
    # yidx_for_upsampled_array = (2*y_idxs_2d)%(2*ny)
    xidx_for_upsampled_array = (2*x_idxs_2d - p_array)%(2*nx)
    yidx_for_upsampled_array = (2*y_idxs_2d - q_array)%(2*ny)

    # xidx_for_upsampled_array = (2*x_idxs_2d - 2*p_array)%(2*nx)
    # yidx_for_upsampled_array = (2*y_idxs_2d - 2*q_array)%(2*ny)

    vchiresidual_x = vx_array - p_array*dx/(2*code_dt)
    vchiresidual_y = vy_array - q_array*dy/(2*code_dt)
    print("xidx_for_upsampled_array = ", xidx_for_upsampled_array)
    print("yidx_for_upsampled_array = ", yidx_for_upsampled_array)
    print("p_array = ", p_array)
    print("q_array = ", q_array)

    # Original (and still the best?)
    # rhs_array = - 2*code_dt*(vchiresidual_x * dgdxold_array[yidx_for_upsampled_array, xidx_for_upsampled_array]
    #         + vchiresidual_y * dgdyold_array[yidx_for_upsampled_array, xidx_for_upsampled_array] )
    # New: an ad-hoc change
    rhs_array = - 2*code_dt*(vchiresidual_x * dgdxold_array[2*y_idxs_2d, xidx_for_upsampled_array]
            + vchiresidual_y * dgdyold_array[yidx_for_upsampled_array, 2*x_idxs_2d] )
    gnew_array_python = golder_array[yidx_for_norm_array, xidx_for_norm_array] + rhs_array
    gnew_slonly_array_python = golder_array[yidx_for_norm_array, xidx_for_norm_array]
    print("Done")

    return gnew_slonly_array_python, gnew_array_python

def get_array_from_file(file_longname, padding_val=1):
    """ """
    my_array = np.zeros((padding_val*ny,padding_val*nx))
    myfile = open(file_longname, "r")
    filetext = (myfile.read()).strip()
    myfile.close()
    for line_idx, line in enumerate(re.split("\n", filetext)):
        entries = re.split("\s+", line.strip())
        for entry_idx, entry in enumerate(entries):
            my_array[line_idx, entry_idx] = float(entry)

    return my_array

if __name__ == "__main__":

    plaintext_folder = "single_mode_nisl_text_files/"
    golder_file = plaintext_folder + "golderxy.txt"
    gnew_sl_only_file = plaintext_folder + "gnewxy_no_rhs.txt"
    gnew_file = plaintext_folder + "gnewxy.txt"
    gnew_tracing_traj_file = plaintext_folder + "gnewxy_tracing_traj.txt"

    dgolddy_file = plaintext_folder + "dgolddy.txt"
    dgolddx_file = plaintext_folder + "dgolddx.txt"
    vx_file = plaintext_folder + "vchiold_x.txt"
    vy_file = plaintext_folder + "vchiold_y.txt"

    golder_array = get_array_from_file(golder_file)
    gnew_slonly_array_stella = get_array_from_file(gnew_sl_only_file)
    gnew_array_stella = get_array_from_file(gnew_file)
    gnew_array_tracking_traj_stella = get_array_from_file(gnew_tracing_traj_file)

    vx_array = get_array_from_file(vx_file)
    vy_array = get_array_from_file(vy_file)
    dgolddy_array = get_array_from_file(dgolddy_file, padding_val=2)
    dgolddx_array = get_array_from_file(dgolddx_file, padding_val=2)

    gnew_slonly_array_python, gnew_array_python = visualise_nisl_calculation(golder_array, dgolddx_array, dgolddy_array, vx_array, vy_array)

    plot_arrays_line_by_line_from_files(golder_array, gnew_slonly_array_stella,
                                        gnew_array_stella, gnew_array_tracking_traj_stella)
    # plot_arrays_line_by_line_compare_to_python(golder_array, gnew_slonly_array_stella, gnew_array_stella,
    #                                            gnew_slonly_array_python, gnew_array_python)
