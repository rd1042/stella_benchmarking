"""Toy code to test interpolation using lagrangian polynomials"""


import numpy as np
from scipy.interpolate import lagrange, interp1d, interp2d
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


def calculate_polynomials_explicitly():
    """ """
    x = [0, 1, 2]
    y = [1, 3, 2]
    P1_coeff = [1,-1.5,.5]
    P2_coeff = [0, 2,-1]
    P3_coeff = [0,-.5,.5]

    # get the polynomial function
    P1 = poly.Polynomial(P1_coeff)
    P2 = poly.Polynomial(P2_coeff)
    P3 = poly.Polynomial(P3_coeff)

    x_new = np.arange(-1.0, 3.1, 0.1)

    fig = plt.figure(figsize = (10,8))
    plt.plot(x_new, P1(x_new), 'b', label = 'P1')
    plt.plot(x_new, P2(x_new), 'r', label = 'P2')
    plt.plot(x_new, P3(x_new), 'g', label = 'P3')
    plt.scatter(x,y, c="black")
    plt.plot(x, np.ones(len(x)), 'ko', x, np.zeros(len(x)), 'ko')
    plt.title('Lagrange Basis Polynomials')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()

    return

def use_scipy_lagrange():
    """ """
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    y = [1, 3, 2, 2, 3, 4, 2, 1, 0, -1, 1, 2,   4,  8, 10, 11, 14, 18, 20, 26, 35, 42, 50, 64, 72, 78, 76, 61, 54, 46, 43]

    lagrange_fit = lagrange(x,y)

    x_new = np.arange(-1.0, 31.1, 0.1)

    fig = plt.figure(figsize = (10,8))
    plt.plot(x_new, lagrange_fit(x_new), 'b', label = 'P1')
    plt.scatter(x,y, c="black")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()


def test_cubic_interpolation():
    """ """
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 3, 2, 2, 3, 4, 2, 1, 0, -1, 1]

    linear_fit = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    cubic_fit = interp1d(x, y, kind="cubic", bounds_error=False, fill_value="extrapolate")


    x_new = np.arange(0, 11.1, 0.1)

    fig = plt.figure(figsize = (10,8))
    plt.plot(x_new, linear_fit(x_new), label = 'linear')
    plt.plot(x_new, cubic_fit(x_new), label = 'cubic')
    plt.scatter(x,y, c="black")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Hello world!")
    # use_scipy_lagrange()
    test_cubic_interpolation()
