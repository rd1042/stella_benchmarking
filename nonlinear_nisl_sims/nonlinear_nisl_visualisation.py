""" """


import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import math

nonlinear_file = "input_leapfrog_restarted.nonlinear_quantities"
naky = 5
nakx = 7
ny = 14
nx = 10

dx = 1.9230769230769229
dy = 1.3463968515384828
dt = 0.134
xmax = (nx) * dx
ymax = (ny) * dy

ky = np.array([0.0000000000000000, 0.33333333333333331, 0.66666666666666663, 1.0000000000000000, 1.3333333333333333])
kx = np.array([0.0000000000000000, 0.32672563597333848, 0.65345127194667696,
                0.98017690792001544, -0.98017690792001544, -0.65345127194667696, -0.32672563597333848])

x_grid = np.arange(0, xmax, dx )
y_grid = np.arange(0, ymax, dy )
x_idxs = np.arange(0, nx, 1, dtype="int")
y_idxs = np.arange(0, ny, 1, dtype="int")

def get_array_for_real_plaintext(plaintext_block):
    """ """
    #print("plaintext_block = ", plaintext_block)
    # Get rid of "quantity = " term
    cropped_plaintext = re.split(":", plaintext_block.strip())[1]
    lines_str_list = re.split("\n", cropped_plaintext.strip())

    data_array = np.zeros((ny, nx))
    for iy in range(0, ny):
        entry_strs = re.split("\s+", lines_str_list[iy].strip())
        for ix in range(0, nx):
            data_array[iy,ix] = float(entry_strs[ix])

    #print("data_array = ", data_array)

    return data_array

def get_array_for_complex_plaintext(plaintext_block):
    """ """
    cropped_plaintext = re.split(":", plaintext_block.strip())[1]
    # Each line is a single iy i.e. coresponding to a single y value
    lines_str_list = re.split("\n", cropped_plaintext.strip())

    data_array = np.zeros((naky, nakx), dtype="complex")
    for iky in range(0, naky):
        entry_strs = re.split("\s+", lines_str_list[iky].strip())
        for ikx in range(0, nakx):
            [real_str, imag_str] = re.split(",", entry_strs[ikx])
            real_val = float(real_str[1:])
            imag_val = float(imag_str[:-1])
            data_array[iky,ikx] = real_val + 1j*imag_val


    return data_array

def get_arrays_from_nonlinear_data():
    """ """
    myfile = open(nonlinear_file, "r")
    data = myfile.read()
    myfile.close()
    data_blocks = re.split("XXXXXXXXXXXXXXXXXXXXXX", data.strip())
    golder_block = data_blocks[0]
    golderyx_block = data_blocks[1]
    dgold_dy_block = data_blocks[2]
    dgold_dx_block = data_blocks[3]
    vchiold_y_block = data_blocks[4]
    vchiold_x_block = data_blocks[5]

    golderyx_array = get_array_for_real_plaintext(golderyx_block)
    dgold_dy_array = get_array_for_real_plaintext(dgold_dy_block)
    dgold_dx_array = get_array_for_real_plaintext(dgold_dx_block)
    vchiold_y_array = get_array_for_real_plaintext(vchiold_y_block)
    vchiold_x_array = get_array_for_real_plaintext(vchiold_x_block)
    golder_array = get_array_for_complex_plaintext(golder_block)

    return [golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
        vchiold_y_array, vchiold_x_array]


def create_upsamled_grid(data_array):
    """Given a data array, return an upsampled version of the array.
    Original array samples on an x, y grid:
    *   *   *   *   *

    *   *   *   *   *

    *   *   *   *   *

    We want to sample at midpoints:
    * & * & * & * & * &
    & & & & & & & & & &
    * & * & * & * & * &
    & & & & & & & & & &
    * & * & * & * & * &
    & & & & & & & & & &

    where & denotes a new sampling point. The final & in x, y should be an
    interpolation of the final * and the first * (periodic).
    """
    # print("data_array.shape = ", data_array.shape)
    data_array_upsampled = np.zeros((2*ny, 2*nx))

    ## Probably can do this with scipy's interpolate library, but
    ## want to ensure BCs are correct.
    total_upsampled_gridpoints = 4 * nx*ny
    upsampled_x_idx = 0
    upsampled_y_idx = 0

    for upsampled_xidx in range(0, 2*nx):
        for upsampled_yidx in range(0, 2*ny):
            ## 4 cases to consider:
            # (1) x_idx even, y_idx even. Use the original data point
            # (2) x_idx even, y_idx odd. Upsampled value is 1/2 y_up, 1/2 y_down
            # (3) x_idx odd, y_idx even. Upsampled value is 1/2 x_left, 1/2 x_right
            # (4) x_idx odd, y_idx odd. Upsampled value is 1/4 (x_left,y_up),
            #                            1/4 (x_left,y_down), 1/4 (x_right,y_up), 1/4 (x_right,y_down)
            if (upsampled_xidx%2 == 0) and (upsampled_yidx%2 == 0):
                data_array_upsampled[upsampled_yidx, upsampled_xidx] = data_array[int(upsampled_yidx/2), int(upsampled_xidx/2)]
            elif (upsampled_xidx%2 == 0) and (upsampled_yidx%2 != 0):
                yidx_down = math.floor(upsampled_yidx/2)
                # %(ny) means that the final upsampled_yidx  uses yidx_up = 0
                yidx_up = math.ceil(upsampled_yidx/2)%(ny)
                data_array_upsampled[upsampled_yidx, upsampled_xidx] = (0.5*data_array[yidx_down, int(upsampled_xidx/2)]
                                                        +   0.5*data_array[yidx_up, int(upsampled_xidx/2)])
            elif (upsampled_xidx%2 != 0) and (upsampled_yidx%2 == 0):
                xidx_left = math.floor(upsampled_xidx/2)
                # %(nx) means that the final upsampled_xidx  uses xidx_right = 0
                xidx_right = math.ceil(upsampled_xidx/2)%(nx)
                data_array_upsampled[upsampled_yidx, upsampled_xidx] = (0.5*data_array[int(upsampled_yidx/2), xidx_left]
                                                        +   0.5*data_array[int(upsampled_yidx/2), xidx_right])

            elif (upsampled_xidx%2 != 0) and (upsampled_yidx%2 != 0):
                xidx_left = math.floor(upsampled_xidx/2)
                # %(nx) means that the final upsampled_xidx  uses xidx_right = 0
                xidx_right = math.ceil(upsampled_xidx/2)%(nx)
                yidx_down = math.floor(upsampled_yidx/2)
                # %(ny) means that the final upsampled_yidx  uses yidx_up = 0
                yidx_up = math.ceil(upsampled_yidx/2)%(ny)
                data_array_upsampled[upsampled_yidx, upsampled_xidx] = (0.25*data_array[yidx_down, xidx_left]
                                                        +   0.25*data_array[yidx_up, xidx_left]
                                                        +   0.25*data_array[yidx_down, xidx_right]
                                                        +   0.25*data_array[yidx_up, xidx_right])

    return data_array_upsampled

def nisl_step(golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
                vchiold_y_array, vchiold_x_array):
    """ """
    #######################################################################
    # gnew(x_i, y_j) = golder[x_i - p_ij*dx, y_j - * q_ij*dy]
    #                       + rhs_ij
    #  with
    #
    #  rhs_ij = - (vresidual_x_ij*dgold_dx[x_i - p_ij*dx/2, y_j - * q_ij*dy/2]
    #            + vresidual_y_ij*dgold_dy[x_i - p_ij*dx/2, y_j - * q_ij*dy/2])
    #
    # vresidual_x_ij = vchiold_x[x_i - p_ij*dx/2, y_j - * q_ij*dy/2]
    #                    - p_ij * dx/(2*dt)
    # vresidual_y_ij = vchiold_y[x_i - p_ij*dx/2, y_j - * q_ij*dy/2]
    #                    - q_ij * dy/(2*dt)
    #
    #######################################################################
    ### OLD IDEA
    # Estimate the departure points based on vchiold
    # #####################################################################
    # Idea: Perform
    # (a) p_ij = q_ij = 0
    # (b) p_ij = NINT(2 dt/dx vchiold_x[x_i - p*dx/2, y_j - q*dy/2])
    #     q_ij = NINT(2 dt/dy vchiold_y[x_i - p*dx/2, y_j - q*dy/2])
    # (c) Repeat (b) a few times
    #
    # To get vchiold_x,y[(x_i - p*dx/2), (y_j - q*dy/2)], need to upsample
    # vchiold_x,y
    # #####################################################################

    ### New idea
    # Idea: instead of
    #
    # print("x_grid = ", x_grid)
    # print("y_grid = ", y_grid)
    p_array = np.zeros((ny, nx), dtype="int")
    q_array = np.zeros((ny, nx), dtype="int")

    # Upsample - double the number of points in the vchi, gold, golder grids.
    golderyx_array_upsampled = create_upsamled_grid(golderyx_array)
    # # fig = plt.figure()
    # # plt.show()
    # sys.exit()
    dgold_dy_array_upsampled = create_upsamled_grid(dgold_dy_array)
    dgold_dx_array_upsampled = create_upsamled_grid(dgold_dx_array)
    vchiold_x_array_upsampled = create_upsamled_grid(vchiold_x_array)
    vchiold_y_array_upsampled = create_upsamled_grid(vchiold_y_array)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dgold_dy_array)
    ax2.imshow(dgold_dy_array_upsampled)
    ax1.set_ylabel("x idx")
    ax2.set_ylabel("x idx")
    ax1.set_xlabel("y idx")
    ax2.set_xlabel("y idx")
    plt.show()

    upsampled_xidxs = np.arange(0, 2*nx, 1, dtype="int")
    upsampled_yidxs = np.arange(0, 2*ny, 1, dtype="int")
    usampled_xidxs_2d = np.zeros((2*ny, 2*nx))
    usampled_yidxs_2d = np.zeros((2*ny, 2*nx))

    # Update p, q;
    # (b) p_ij = NINT(2 dt/dx vchiold_x[x_i - p*dx/2, y_j - q*dy/2])
    #     q_ij = NINT(2 dt/dy vchiold_y[x_i - p*dx/2, y_j - q*dy/2])

    # The idxs are different between the "normal" (i.e. non-upsampled) arrays
    # and the upsampled arrays. Get the correct upsampled idx for
    # (x_i - p*dx/2), (y_j - q*dy/2) here.
    xidx_array = upsampled_xidxs - p_array
    yidx_array = upsampled_yidxs - q_array
    print("xidx_array = ", xidx_array)

    # Find p_ij, q_ij again, check if the values have changed


    # Calculate the residual velocities

    # Calculate rhs_ij

    # Calculate gnew

    return


if __name__ == "__main__":

    [golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
        vchiold_y_array, vchiold_x_array] = get_arrays_from_nonlinear_data()#

    nisl_step(golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
                vchiold_y_array, vchiold_x_array)
    # print("len(golder_array) = ", len(golder_array))
    # print("len(golderyx_array) = ", len(golderyx_array))
    # print("golder_array = ", golder_array)
    # fig = plt.figure()
    # plt.imshow(abs(golderyx_array))
    # plt.show()
