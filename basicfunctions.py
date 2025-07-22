import numpy as np
import scipy as sp
def supremum(f,a,b, precision=10.**(-5)) : #Trouve approx le sup d'une fonction f sur [a,b]
    """Returns an approximation of the supremum of a function on an interval
    params:
        f : function to evaluate
        a : float, lower bound of the interval
        b: float, upper bound of the interval
        precision: float, tolerated argmax error
    """
    X = np.arange(a,b, precision)
    return max(f(X))

def exponential_kernel(x,alpha,beta) :
    return alpha*np.exp(-beta*x)

def cosinusoidal_kernel(x,alpha,beta) :
    return np.where(
        (beta * x >= 0) & (beta * x < np.pi / 2),
        alpha * np.cos(beta * x),
        0
    )

def sinusoidal_kernel(x,alpha,beta) :
    return np.where(
        (beta * x >= 0) & (beta * x < np.pi),
        alpha * np.sin(beta * x),
        0
    )


def n_convolution(phi, n, start_interval, end_interval, resolution=1e-3):
    """
    Compute the n-fold convolution of function phi with itself over [start_interval, end_interval].

    :param phi: function to be convolved with itself
    :param n: number of convolutions
    :param start_interval: start of the time interval
    :param end_interval: end of the time interval
    :param resolution: step size for discretization (default: 1e-4)
    :return: dictionary with 'convolution_table' and 'x_axis'
    """
    assert n > 0

    dt = resolution
    x_axis = np.arange(start_interval, end_interval + dt, dt)
    phi_vals = phi(x_axis)

    current_conv = phi_vals.copy()
    for _ in range(1, n):
        current_conv = np.convolve(current_conv, phi_vals) * dt

    conv_x_axis = np.arange(0, dt * len(current_conv), dt)

    return {
        "convolution_table": current_conv,
        "x_axis": conv_x_axis
    }

def n_convolution_t(t, conv_array):
    """
    Evaluate the n-fold convolution at a given time t using linear interpolation.

    :param t: time at which to evaluate the convolution
    :param conv_array: output dictionary from n_convolution
    :return: interpolated convolution value at time t
    """
    conv = conv_array["convolution_table"]
    x_axis = conv_array["x_axis"]

    if t <= x_axis[0]:
        return conv[0]
    elif t >= x_axis[-1]:
        return conv[-1]
    else:
        return np.interp(t, x_axis, conv)

if __name__=="__main__":
    alpha= 1
    beta = 2.5
    T=5

    absc= np.linspace(0,beta*T)
    import matplotlib.pyplot as plt
    plt.plot(absc, cosinusoidal_kernel(absc, alpha,beta))
    plt.plot(absc, sinusoidal_kernel(absc, alpha, beta))

    plt.show()