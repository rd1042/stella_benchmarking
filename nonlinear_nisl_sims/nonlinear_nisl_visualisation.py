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
x_grid_upsampled = np.arange(0, xmax, dx/2 )
y_grid_upsampled = np.arange(0, ymax, dy/2 )
x_idxs = np.arange(0, nx, 1, dtype="int")
y_idxs = np.arange(0, ny, 1, dtype="int")

## Get the idxs in 2D
x_idxs_2d = np.zeros((ny,nx), dtype="int")
y_idxs_2d = np.zeros((ny,nx), dtype="int")
x_grid_2d = np.zeros((ny,nx))
y_grid_2d = np.zeros((ny,nx))
x_grid_2d_upsampled = np.zeros((2*ny,2*nx))
y_grid_2d_upsampled = np.zeros((2*ny,2*nx))


for yidx in range(0, ny):
    x_idxs_2d[yidx, :] = x_idxs
    x_grid_2d[yidx, :] = x_grid

for xidx in range(0, nx):
    y_idxs_2d[:,xidx] = y_idxs
    y_grid_2d[:,xidx] = y_grid

for yidx in range(0, 2*ny):
    x_grid_2d_upsampled[yidx, :] = x_grid_upsampled

for xidx in range(0, 2*nx):
    y_grid_2d_upsampled[:,xidx] = y_grid_upsampled
#
# print("x_grid = ", x_grid)
# print("y_grid = ", y_grid)
# print("x_grid_2d = ", x_grid_2d)
# print("y_grid_2d = ", y_grid_2d)

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


def create_upsampled_grid(data_array):
    """Given a data array, return an upsampled version of the array.
    Original array samples on an x, y grid:
    *   *   *   *   *

    *   *   *   *   *

    *   *   *   *   *

    We want to sample at midpoints:
    & & & & & & & & & &
    * & * & * & * & * &
    & & & & & & & & & &
    * & * & * & * & * &
    & & & & & & & & & &
    * & * & * & * & * &

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

def nisl_step_ritchie(golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
                vchiold_y_array, vchiold_x_array):
    """The NISL step according to Ritchie; here we don't calculate the (approximate)
    departure point, but instead (attempt to) find the trajectory from t^n which arrives
    closest to our gridpoint at t^(n+1)"""

    def update_p_and_q(p_array, q_array, vchiold_x_array_upsampled, vchiold_y_array_upsampled,
                        yidx_for_upsampled_array, xidx_for_upsampled_array):
        """Update guess for p, q, and the idxs corresponding to the upsampled
        array."""
        p_array = np.rint(2 * dt/dx * vchiold_x_array_upsampled[yidx_for_upsampled_array, xidx_for_upsampled_array]).astype("int")
        q_array = np.rint(2 * dt/dy * vchiold_y_array_upsampled[yidx_for_upsampled_array, xidx_for_upsampled_array]).astype("int")
        xidx_for_upsampled_array = (2*x_idxs_2d - p_array)%(2*nx)
        yidx_for_upsampled_array = (2*y_idxs_2d - q_array)%(2*ny)
        return p_array, q_array, yidx_for_upsampled_array, xidx_for_upsampled_array

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







    # p=0 means shift non-upsampled xidx by 0, and shift upsampled xidx by 0
    # p=1 means shift non-upsampled xidx by 1, and shift upsampled xidx by 1
    #     (because upsampled x is being shifted by p*dx/2, but is sampled
    #      every dx/2)


    p_array = np.zeros((ny, nx), dtype="int")
    q_array = np.zeros((ny, nx), dtype="int")

    # Upsample - double the number of points in the vchi, gold, golder grids.
    # golderyx_array_upsampled = create_upsampled_grid(golderyx_array)

    dgold_dy_array_upsampled = create_upsampled_grid(dgold_dy_array)
    dgold_dx_array_upsampled = create_upsampled_grid(dgold_dx_array)
    vchiold_x_array_upsampled = create_upsampled_grid(vchiold_x_array)
    vchiold_y_array_upsampled = create_upsampled_grid(vchiold_y_array)
    # To get the arrival points:
    # xnew = (xold + (2*dt*vchiold_x))%xmax
    # ynew = (yold + (2*dt*vchiold_y))%ymax

    # As a diagnostic, plot the regular grid and the arrival locations.
    marker_size = 20.

    xnew = (x_grid_2d_upsampled - (dt*vchiold_x_array_upsampled))%xmax
    ynew = (y_grid_2d_upsampled - (dt*vchiold_y_array_upsampled))%ymax

    fig = plt.figure(figsize=[12, 8])
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_grid_2d_upsampled.flatten(), y_grid_2d_upsampled.flatten(), marker="x", s=marker_size, label="upsampled grid")
    ax1.scatter(x_grid_2d.flatten(), y_grid_2d.flatten(), s=60, label="grid")
    ax1.scatter(xnew.flatten(), ynew.flatten(), s=marker_size, label="arrival points")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.grid(True)
    ax1.set_xlim([-1, 23])
    ax1.legend(loc="upper right")
    plt.show()
    # sys.exit()


    upsampled_xidxs = np.arange(0, 2*nx, 1, dtype="int")
    upsampled_yidxs = np.arange(0, 2*ny, 1, dtype="int")
    # usampled_xidxs_2d = np.zeros((2*ny, 2*nx))
    # usampled_yidxs_2d = np.zeros((2*ny, 2*nx))

    # Update p, q;
    # (b) p_ij = NINT(2 dt/dx vchiold_x[x_i - p*dx/2, y_j - q*dy/2])
    #     q_ij = NINT(2 dt/dy vchiold_y[x_i - p*dx/2, y_j - q*dy/2])

    # The idxs are different between the "normal" (i.e. non-upsampled) arrays
    # and the upsampled arrays. Get the correct upsampled idx for
    # (x_i - p*dx/2), (y_j - q*dy/2) here.
    #
    # Each (x,y) gridpoint has an integer value of phalf and qhalf; from this
    # we want to get the idxs fro the upsampled quantities; the t^n quantities
    # have twice as many gridpoints, so for a particular (xidx, yidx, phalf, qhalf)
    # the corresponding idxs of the upsampled data points are
    # xidx_for_upsampled_data = 2*(xidx - pidx/2) = 2*xidx - pidx
    # yidx_for_upsampled_data = 2*(yidx - qidx/2) = 2*yidx - qidx
    yidx = 5 ; xidx = 6

    xidx_for_upsampled_array = (2*x_idxs_2d - p_array)%(2*nx)
    yidx_for_upsampled_array = (2*y_idxs_2d - q_array)%(2*ny)
    # We've got the idxs - update p, q
    for counter in range(0, 1000, 1):
        #print("counter = ", counter)
        p_array, q_array, yidx_for_upsampled_array, xidx_for_upsampled_array = update_p_and_q(
                    p_array, q_array, vchiold_x_array_upsampled, vchiold_y_array_upsampled,
                    yidx_for_upsampled_array, xidx_for_upsampled_array)
        # print("xidx_for_upsampled_array[yidx, xidx], yidx_for_upsampled_array[yidx, xidx] = ",
        #         xidx_for_upsampled_array[yidx, xidx], yidx_for_upsampled_array[yidx, xidx])
        # print("p_array[yidx, xidx], q_array[yidx, xidx] = ", p_array[yidx, xidx], q_array[yidx, xidx])

    xidx_for_norm_array = (x_idxs_2d - p_array)%nx
    yidx_for_norm_array = (y_idxs_2d - q_array)%ny
    print("p_array = ", p_array)  # For the stella-given quantities, vchi small so p=q=0 everywhere.
    print("q_array = ", q_array)

    # Calculate the residual velocities
    vchiresidual_x = vchiold_x_array_upsampled[yidx_for_upsampled_array, xidx_for_upsampled_array] - p_array*dx/(2*dt)
    vchiresidual_y = vchiold_y_array_upsampled[yidx_for_upsampled_array, xidx_for_upsampled_array] - q_array*dy/(2*dt)

    # print("vchiold_x, p, q, p_array*dx/(2*dt), vchiresidual_x = ",
    #         vchiold_x_array_upsampled[yidx_for_upsampled_array[yidx,xidx], xidx_for_upsampled_array[yidx,xidx]],
    #         p_array[yidx, xidx], q_array[yidx, xidx], p_array[yidx, xidx]*dx/(2*dt),
    #         vchiresidual_x[yidx, xidx])

    # for yidx in y_idxs:
    #     for xidx in x_idxs:
    #
    #         print("vchiold_x, p, q, p_array*dx/(2*dt), vchiresidual_x = ",
    #                 vchiold_x_array_upsampled[yidx_for_upsampled_array[yidx,xidx], xidx_for_upsampled_array[yidx,xidx]],
    #                 p_array[yidx, xidx], q_array[yidx, xidx], p_array[yidx, xidx]*dx/(2*dt),
    #                 vchiresidual_x[yidx, xidx])
    Courant_num_array = (vchiold_x_array*dt/dx + vchiold_y_array*dt/dy)
    Courant_residual_array = (vchiresidual_x*dt/dx + vchiresidual_y*dt/dy)
    print("max Courant no = ", np.max(abs(Courant_num_array)))
    print("max residual Courant no = ", np.max(abs(Courant_residual_array)))
    #print("dx/dt, max(vchiresidual_x), dy/dt, max(vchiresidual_y) = ", dx/dt, np.max(vchiresidual_x), dy/dt, np.max(vchiresidual_y))
    # Calculate rhs_ij
    #  rhs_ij = - (vresidual_x_ij*dgold_dx[x_i - p_ij*dx/2, y_j - * q_ij*dy/2]
    #            + vresidual_y_ij*dgold_dy[x_i - p_ij*dx/2, y_j - * q_ij*dy/2])
    rhs_array = - (vchiresidual_x * dgold_dy_array_upsampled[yidx_for_upsampled_array, xidx_for_upsampled_array]
                    + vchiresidual_y * dgold_dx_array_upsampled[yidx_for_upsampled_array, xidx_for_upsampled_array] )

    # Calculate gnew
    gnewyx_array = golderyx_array[yidx_for_norm_array, xidx_for_norm_array] + rhs_array

    #print("gnew = ", gnew)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.imshow(dgold_dy_array)
    # ax2.imshow(dgold_dy_array_upsampled)
    # ax1.set_ylabel("x idx")
    # ax2.set_ylabel("x idx")
    # ax1.set_xlabel("y idx")
    # ax2.set_xlabel("y idx")
    # plt.show()
    #
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(golderyx_array)
    ax2.imshow(gnewyx_array)
    ax1.set_ylabel("x idx")
    ax2.set_ylabel("x idx")
    ax1.set_xlabel("y idx")
    ax2.set_xlabel("y idx")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(y_grid, golderyx_array[:,5])
    ax1.plot(y_grid, gnewyx_array[:,5])
    print("x_grid[5] = ", x_grid[5] )
    ax1.set_ylabel("g")
    ax1.set_xlabel("y")
    plt.show()

    return

def nisl_step_finding_departure_point(golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
                vchiold_y_array, vchiold_x_array):
    """Take a NISL step, but finding the "actual" (approximate) departure point.
    Guaranteed to have good stability properties """

    very_small_dt = 1e-8

    p_array = np.zeros((ny, nx), dtype="int")
    q_array = np.zeros((ny, nx), dtype="int")

    # Upsample - double the number of points in the vchi, gold, golder grids.
    # golderyx_array_upsampled = create_upsampled_grid(golderyx_array)

    dgold_dy_array_upsampled = create_upsampled_grid(dgold_dy_array)
    dgold_dx_array_upsampled = create_upsampled_grid(dgold_dx_array)
    vchiold_x_array_upsampled = create_upsampled_grid(vchiold_x_array)
    vchiold_y_array_upsampled = create_upsampled_grid(vchiold_y_array)


    def get_approx_departure_point(yidx, xidx):
        """ """
        upsampled_xidx = 2*xidx
        upsampled_yidx = 2*yidx

        # location of the gridpoint (=arrival point)
        x = x_grid[xidx]; y=y_grid[yidx]
        # Normalised values; these are integer at cell boundaries.
        time_remaining = 2*dt
        xhistory = []
        yhistory = []
        xhistory.append(x); yhistory.append(y)

        # Velocities in the initial cell - put here rather than in the loop,
        # because need to update velocities carefully (the "divergent velocities")
        # problem.
        # NB these are the velocities of the cells, but we'll actually be moving
        # back in time.
        u_x = vchiold_x_array_upsampled[upsampled_yidx, upsampled_xidx]
        u_y = vchiold_y_array_upsampled[upsampled_yidx, upsampled_xidx]

        max_iterations = max((100*np.mean(abs(vchiold_x_array_upsampled))*dt/dx) , (100*np.mean(abs(vchiold_y_array_upsampled))*dt/dy))
        #print("max_iterations = ", max_iterations)
        counter=0
        while time_remaining > 0 and counter < max_iterations:
            counter +=1
            ### TODO: Find something sensible to do if the velocities are
            ### divergent (pushing between boxes)
            # Velocity with which we move (although we'll actually be moving
            # back in time.)

            # Take a step in (x,y) based on the velocity at [x, y, t^n].
            xnorm = (2*x/dx - 0.5); ynorm = (2*y/dy - 0.5)
            # We don't %xmax, ymax here because then we'd have to worry about the
            # sign of <>dist_dt
            if u_x > 0 :
                xboundary = ((np.floor(xnorm) + 0.5) * dx/2)
            else:
                xboundary = ((np.ceil(xnorm)  + 0.5) * dx/2)
            if u_y > 0 :
                yboundary = ((np.floor(ynorm) + 0.5) * dy/2)
            else:
                yboundary = ((np.ceil(ynorm)  + 0.5) * dy/2)

            # print("upsampled_xidx, upsampled_yidx = ", upsampled_xidx, upsampled_yidx )
            # print("xnorm, u_x, xboundary, ynorm, u_y, yboundary = ", xnorm, u_x, xboundary, ynorm, u_y, yboundary)
            # Calculate the time required to reach the nearest boundaries in y and x
            # Careful though - if our velocities are too small we'll get <>dist_dt=inf
            min_vx_magnitude = (dx/dt)*1e-10   # We know for sure that the trajectory won't make it into the next cell
            min_vy_magnitude = (dy/dt)*1e-10   # We know for sure that the trajectory won't make it into the next cell

            if abs(u_y) < min_vy_magnitude:
                ydist_dt = 10*dt    # This ensures ydist_dt > time_remaining
            else:
                ydist_dt = -(yboundary - y)/u_y
            if abs(u_x) < min_vx_magnitude:
                xdist_dt = 10*dt # This ensures xdist_dt > time_remaining
            else:
                xdist_dt = -(xboundary - x)/u_x

            #print("ydist_dt, xdist_dt = ", ydist_dt, xdist_dt )

            if (ydist_dt > time_remaining) and (xdist_dt > time_remaining):
                #print("STOPPING before next boundary")
                # Update location
                y = (y - u_y * time_remaining)%ymax
                x = (x - u_x * time_remaining)%xmax
                time_remaining = 0
            else:
                if ydist_dt < xdist_dt:
                    #print("Hit y!")
                    # Hit the boundary in y
                    y = (yboundary - u_y*very_small_dt )%ymax   # slightly overstep, so we're definitely in the next cell
                    x = (x - u_x * (ydist_dt + very_small_dt))%xmax

                    # Update the values of u_x, u_y
                    if u_y > 0:
                        # +ve u_y, so going back in time we're going in the -ve y direction; our new cell is
                        # below the old cell
                        upsampled_yidx = (upsampled_yidx - 1)%(2*ny)
                    else:
                        # -ve u_y, so going back in time we're going in the +ve y direction; our new cell is
                        # below the old cell
                        upsampled_yidx = (upsampled_yidx + 1)%(2*ny)

                    # Update the velocities
                    u_x = vchiold_x_array_upsampled[upsampled_yidx, upsampled_xidx]
                    # Update u_y, but if the sign is different, we're going to
                    # bounce back and forth (this indicates the velocity falling
                    # to zero somewhere between the 2 cell centres). To avoid the "bouncing",
                    # set velocity to zero.
                    u_y_new = vchiold_y_array_upsampled[upsampled_yidx, upsampled_xidx]
                    if (u_y * u_y_new) < 0:
                        # Opposite signs, so set u_y=0
                        u_y=0
                    else:
                        u_y = u_y_new

                    # Update time_remaining. Include the "very small dt" contribution
                    time_remaining = time_remaining - (ydist_dt + very_small_dt)
                else:
                    #print("Hit x!")
                    # Hit the boundary in x
                    x = (xboundary - u_x*very_small_dt)%xmax    # slightly overstep, so we're definitely in the next cell
                    y = (y - u_y * (xdist_dt + very_small_dt))%ymax

                    # Update the values of u_x, u_y
                    if u_x > 0:
                        # +ve u_x, so going back in time we're going in the -ve x direction; our new cell is
                        # to the left of the old cell
                        upsampled_xidx = (upsampled_xidx - 1)%(2*nx)
                    else:
                        # -ve u_y, so going back in time we're going in the +ve y direction; our new cell is
                        # below the old cell
                        upsampled_xidx = (upsampled_xidx + 1)%(2*nx)

                    # Update velocities
                    u_y = vchiold_y_array_upsampled[upsampled_yidx, upsampled_xidx]
                    # Update u_x, but if the sign is different, we're going to
                    # bounce back and forth (this indicates the velocity falling
                    # to zero somewhere between the 2 cell centres). To avoid the "bouncing",
                    # set velocity to zero.
                    u_x_new = vchiold_x_array_upsampled[upsampled_yidx, upsampled_xidx]
                    if (u_x * u_x_new) < 0:
                        # Opposite signs, so set u_y=0
                        u_x=0
                    else:
                        u_x = u_x_new

                    # Update time_remaining. Include the "very small dt" contribution
                    time_remaining = time_remaining - (xdist_dt + very_small_dt)

                    # Check whether we've been in the

            xhistory.append(x); yhistory.append(y)

        ########################################################################
        ##### DIAGNOSTIC PLOTS #################################################
        ########################################################################
        def basic_diagnostic_plot_trajectories():
            """The "vanilla" plot - show paths and gridpoints. """
            marker_size = 20.
            fig = plt.figure(figsize=[12, 8])
            ax1 = fig.add_subplot(111)
            ax1.scatter(x_grid_2d_upsampled.flatten(), y_grid_2d_upsampled.flatten(), marker="x", s=marker_size, label="upsampled grid")
            ax1.scatter(x_grid_2d.flatten(), y_grid_2d.flatten(), s=60, label="grid")
            ax1.scatter(xhistory, yhistory, s=marker_size, label="trajectory")
            for hist_idx in range(0, len(xhistory)-1):
                ax1.plot([xhistory[hist_idx], xhistory[hist_idx+1]], [yhistory[hist_idx], yhistory[hist_idx+1]])
            ax1.set_xlabel(r"$x$")
            ax1.set_ylabel(r"$y$")
            ax1.grid(True)
            #ax1.set_xlim([-1, 23])
            ax1.legend(loc="upper right")
            plt.show()

        def diagnostic_plot_trajectories():
            """ A more complicated plot - show paths and gridpoints, and boundaries and
            cell velocities."""
            x_grid_upsampled_boundaries = x_grid_upsampled + dx/4
            y_grid_upsampled_boundaries = y_grid_upsampled + dy/4

            # To make the horizontal lines: make 1 horizontal line per
            # y_grid_upsampled_boundaries, starting at x=0 and ending at max(x_grid_upsampled_boundaries)
            horizontal_lines = []
            for diag_yval in y_grid_upsampled_boundaries:
                horizontal_line_xvals = [0, max(x_grid_upsampled_boundaries)]
                horizontal_line_yvals = [diag_yval, diag_yval]
                horizontal_lines.append([horizontal_line_xvals, horizontal_line_yvals])

            # To make the vertical lines: make 1 vertical line per
            # x_grid_upsampled_boundaries, starting at y=0 and ending at max(y_grid_upsampled_boundaries)
            vertical_lines = []
            for diag_xval in x_grid_upsampled_boundaries:
                vertical_line_xvals = [diag_xval, diag_xval]
                vertical_line_yvals = [0, max(y_grid_upsampled_boundaries)]
                vertical_lines.append([vertical_line_xvals, vertical_line_yvals])

            # Normalise velocities such that the largest veloccity occupies a cell length/height.
            # That is, want max(unorm_x) = dx/2 and max(unorm_y) = dy/2
            x_scaling_fac = dx/2 / np.max(abs(vchiold_x_array_upsampled))
            unorm_x = vchiold_x_array_upsampled * x_scaling_fac
            y_scaling_fac = dy/2 / np.max(abs(vchiold_y_array_upsampled))
            unorm_y = vchiold_y_array_upsampled * y_scaling_fac

            # Want to represent these velocities with an arrow, which is centered on
            # the gridpoint. So the starting point of the arrow should be [x - unorm_x/2, y - unorm_y/2]
            arrows = [] # Each item in arrows is a list describing a single arrow; [x, y, delta_x, delta_y]
            for diag_upsampled_xidx in range(0, 2*nx):
                for diag_upsampled_yidx in range(0, 2*ny):
                    arrow_x = x_grid_2d_upsampled[diag_upsampled_yidx, diag_upsampled_xidx] - unorm_x[diag_upsampled_yidx, diag_upsampled_xidx]/2
                    arrow_y = y_grid_2d_upsampled[diag_upsampled_yidx, diag_upsampled_xidx] - unorm_y[diag_upsampled_yidx, diag_upsampled_xidx]/2
                    arrow_dx = unorm_x[diag_upsampled_yidx, diag_upsampled_xidx]
                    arrow_dy = unorm_y[diag_upsampled_yidx, diag_upsampled_xidx]
                    arrows.append([arrow_x, arrow_y, arrow_dx, arrow_dy])


            marker_size = 20.
            arrow_head_width = 0.1
            fig = plt.figure(figsize=[12, 8])
            ax1 = fig.add_subplot(111)
            ax1.scatter(x_grid_2d_upsampled.flatten(), y_grid_2d_upsampled.flatten(), marker="x", s=marker_size, label="upsampled grid")
            ax1.scatter(x_grid_2d.flatten(), y_grid_2d.flatten(), s=60, label="grid")
            ax1.scatter(xhistory, yhistory, s=marker_size, label="trajectory")
            for horizontal_line in horizontal_lines:
                ax1.plot(horizontal_line[0], horizontal_line[1], ls="--", c="gray")
            for vertical_line in vertical_lines:
                ax1.plot(vertical_line[0], vertical_line[1], ls="--", c="gray")
            for arrow in arrows:
                ax1.arrow(arrow[0], arrow[1], arrow[2], arrow[3], color="blue", length_includes_head = True, head_width=arrow_head_width)
            for hist_idx in range(0, len(xhistory)-1):
                ax1.plot([xhistory[hist_idx], xhistory[hist_idx+1]], [yhistory[hist_idx], yhistory[hist_idx+1]])
            ax1.set_xlabel(r"$x$")
            ax1.set_ylabel(r"$y$")
            ax1.grid(True)
            #ax1.set_xlim([-1, 23])
            ax1.legend(loc="upper right")
            plt.show()

            return

        #basic_diagnostic_plot_trajectories()
        #diagnostic_plot_trajectories()
        ########################################################################

        return y, x


    approx_departure_points_x = np.zeros((ny, nx))
    approx_departure_points_y = np.zeros((ny, nx))

    # Find the approximate departure point
    for yidx in range(0, ny):
        for xidx in range(0,nx):#range(0, nx):
            y, x = get_approx_departure_point(yidx, xidx)
            approx_departure_points_x[yidx, xidx] = x
            approx_departure_points_y[yidx, xidx] = y
            #print("y, x = ", y, x)

    ############################################################################
    ### DIAGNSOTIC PLOT
    ############################################################################

    def make_diagnostic_plot_departure_points():
        """ """
        marker_size = 20.
        arrow_head_width = 0.1
        fig = plt.figure(figsize=[12, 8])
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_grid_2d_upsampled.flatten(), y_grid_2d_upsampled.flatten(), marker="x", s=marker_size, label="upsampled grid")
        ax1.scatter(x_grid_2d.flatten(), y_grid_2d.flatten(), s=60, label="grid")
        ax1.scatter(approx_departure_points_x.flatten(), approx_departure_points_y.flatten(), s=marker_size,label="departure point")
        # ax1.scatter(xhistory, yhistory, s=marker_size, label="trajectory")
        # for horizontal_line in horizontal_lines:
        #     ax1.plot(horizontal_line[0], horizontal_line[1], ls="--", c="gray")
        # for vertical_line in vertical_lines:
        #     ax1.plot(vertical_line[0], vertical_line[1], ls="--", c="gray")
        # for arrow in arrows:
        #     ax1.arrow(arrow[0], arrow[1], arrow[2], arrow[3], color="blue", length_includes_head = True, head_width=arrow_head_width)
        # for hist_idx in range(0, len(xhistory)-1):
        #     ax1.plot([xhistory[hist_idx], xhistory[hist_idx+1]], [yhistory[hist_idx], yhistory[hist_idx+1]])
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.grid(True)
        ax1.set_xlim([-0.96, 25])
        #ax1.set_xlim([-1, 23])
        ax1.legend(loc="upper right")
        plt.show()

        return

    # make_diagnostic_plot_departure_points()
    ############################################################################

    # Now we've got approximate departure points, work out p and q.
    #p_array = int(np.rint())


    return

if __name__ == "__main__":

    [golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
        vchiold_y_array, vchiold_x_array] = get_arrays_from_nonlinear_data()#

    ## Let's try artificially increasing vchi to exceed the CFL condition.
    scaling_fac = 10
    vchiold_x_array = vchiold_x_array * scaling_fac
    vchiold_y_array = vchiold_y_array * scaling_fac

    nisl_step_finding_departure_point(golder_array, golderyx_array, dgold_dy_array, dgold_dx_array,
                vchiold_y_array, vchiold_x_array)
    # print("len(golder_array) = ", len(golder_array))
    # print("len(golderyx_array) = ", len(golderyx_array))
    # print("golder_array = ", golder_array)
    # fig = plt.figure()
    # plt.imshow(abs(golderyx_array))
    # plt.show()
