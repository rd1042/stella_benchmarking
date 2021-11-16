""" """

import numpy as np
import matplotlib.pyplot as plt

def plot_amplification_factor_for_rk3ssp_ps():
    """ """
    beta = np.linspace(0,2, 100) # beta = k*U*dt
    g_squared = 1  - beta**4/12 + beta**6/36
    g = np.sqrt(g_squared)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(beta, g)
    ax1.set_xlabel(r"$k\cdot U \cdot \Delta t$")
    ax1.set_ylabel(r"$G$")
    ax1.grid(True)
    plt.show()


def plot_amplification_factor_t_for_leapfrog_rk3_start():
    """ """
    beta = [0.01, 0.1, 0.2, 0.5, 0.8, 1.3, 2, 3]

    nstep = 200
    for beta_idx, beta_val in enumerate(beta):
        # Initialise the array of (complex) G
        g_array = np.zeros((nstep), dtype="complex")

        # First step is with RK3-SSP-PS
        # G = 1 - (ibeta) + (ibeta)^2/2 - (ibeta)^3/6
        #   = 1 - i*beta - beta^2/2 + i*beta^3/6
        g_array[0] = 1 - 1j*beta_val - beta_val*beta_val/2 + 1j*beta_val*beta_val*beta_val/6
        print("abs(g_array[0]) = ", abs(g_array[0]))

        # For the rest of the steps, calculate G in a Leapfrog-y way
        for istep in range(1, nstep):
            g_array[istep] = (1/g_array[istep-1]) - 2j*beta_val

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(1, nstep+1), abs(g_array) )
        ax1.set_xlabel("istep")
        ax1.set_ylabel(r"$\vert G \vert$")
        fig.suptitle(r"$k\cdot U \cdot \Delta t =$" + " {:0.3f}".format(beta_val))
        fig.tight_layout()
        save_name = "{:03d}.png".format(beta_idx)
        plt.savefig(save_name)
        plt.close()

    return

def plot_amplification_factor_for_leapfrog_exact_start():
    """ """

    beta = [0.01, 0.1, 0.2, 0.5, 0.8, 1.3, 2, 3]

    nstep = 200
    for beta_idx, beta_val in enumerate(beta):
        print("beta = ", beta_val)
        # Initialise the array of (complex) G
        g_array = np.zeros((nstep), dtype="complex")

        # First step is with RK3-SSP-PS
        # G = 1 - (ibeta) + (ibeta)^2/2 - (ibeta)^3/6
        #   = 1 - i*beta - beta^2/2 + i*beta^3/6
        g_array[0] = np.exp(1j*beta_val)
        print("abs(g_array[0]) = ", abs(g_array[0]))

        # For the rest of the steps, calculate G in a Leapfrog-y way
        for istep in range(1, nstep):
            g_array[istep] = (1/g_array[istep-1]) - 2j*beta_val

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(1, nstep+1), abs(g_array) )
        ax1.set_xlabel("istep")
        ax1.set_ylabel(r"$\vert G \vert$")
        fig.suptitle(r"$k\cdot U \cdot \Delta t =$" + " {:0.3f}".format(beta_val))
        fig.tight_layout()
        save_name = "{:03d}.png".format(beta_idx)
        plt.savefig(save_name)
        plt.close()

    return


def plot_amplification_factor_t_for_leapfrog_rk3_start_several_rk3_steps():
    """ """
    beta_val = 0.8
    nsteps_rk3_list = [1, 2, 4, 10, 20, 50, 100, 1000]

    nstep = 200
    #for beta_idx, beta_val in enumerate(beta):
    for sim_idx, nsteps_rk3 in enumerate(nsteps_rk3_list):
        nsteps_total = nstep - 1 + nsteps_rk3
        # Initialise the array of (complex) G
        t_array_1 = np.linspace(1, 2, nsteps_rk3, endpoint=False)
        t_array_2 = np.linspace(2, nstep+1, nstep-1, endpoint=False)
        # print("t_array_1 = ", t_array_1)
        # print("t_array_2 = ", t_array_2)
        t_array = np.concatenate((t_array_1, t_array_2))
        # print("t_array = ", t_array)
        g_array = np.zeros((nsteps_total), dtype="complex")

        # First nsteps_rk3 is with RK3-SSP-PS
        # G = 1 - (ibeta) + (ibeta)^2/2 - (ibeta)^3/6
        #   = 1 - i*beta - beta^2/2 + i*beta^3/6
        beta_eff = beta_val/nsteps_rk3
        g_cumulative = 1
        for istep in range(0, nsteps_rk3):
            g_array[istep] = 1 - 1j*beta_eff - beta_eff*beta_eff/2 + 1j*beta_eff*beta_eff*beta_eff/6
            g_cumulative = g_cumulative*g_array[istep]
        print("abs(g_array[0]) = ", abs(g_array[0]))
        print("abs(g_cumulative) = ", abs(g_cumulative))

        # For the rest of the steps, calculate G in a Leapfrog-y way
        g_array[nsteps_rk3] = (1/g_cumulative) - 2j*beta_val
        for istep in range(nsteps_rk3+1, nsteps_total):
            g_array[istep] = (1/g_array[istep-1]) - 2j*beta_val

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(t_array, abs(g_array) )
        ax1.set_xlabel("istep")
        ax1.set_ylabel(r"$\vert G \vert$")
        fig.suptitle(r"$k\cdot U \cdot \Delta t =$" + " {:0.3f}".format(beta_val))
        fig.tight_layout()
        save_name = "nsteps_rk3_{:03d}.png".format(nsteps_rk3)
        plt.savefig(save_name)
        plt.close()

    return




if __name__ == "__main__":
    print("Hello world")
    # plot_amplification_factor_for_rk3ssp_ps()
    # plot_amplification_factor_t_for_leapfrog_rk3_start()
    plot_amplification_factor_for_leapfrog_exact_start()
    # plot_amplification_factor_t_for_leapfrog_rk3_start_several_rk3_steps()
