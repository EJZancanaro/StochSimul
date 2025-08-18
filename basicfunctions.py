import numpy as np
import scipy as sp
from scipy.signal import fftconvolve
import types
import dis

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
    assert np.all(x >= 0)
    assert alpha>0
    assert beta>0
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


def n_convolution(phi, n, x_axis):
    """
    Compute for all i up to n all the the n-fold convolutions of function phi with itself over [start_interval, end_interval]
    Stores it in an array, and returns both the array and the interval of evaluation

    :param phi: function to be convolved with itself
    :param n: number of convolutions
    :x_axis : array of values over which to evaluate the functions
    :return: array of the convolutions up to n
    """
    assert n > 0


    start_interval = x_axis[0]
    grid_size = len(x_axis)
    end_interval = x_axis[-1]

    step = x_axis[1]-x_axis[0]

    array_of_convolutions = np.zeros((n, grid_size))

    phi_vals = phi(x_axis)
    current_conv = phi_vals.copy()
    array_of_convolutions[0] = current_conv
    for i in range(1, n):
        current_conv = step * fftconvolve(current_conv, phi_vals)[:grid_size] #We are not interested in values larger
        array_of_convolutions[i][:] = np.copy(current_conv)                      #than end_interval

        #the multiplication by the step is because the scipy function only computes the sum of the products.
        #which needs to me multiplied by the step to become a riemann approximation of the integral

    return array_of_convolutions


def n_convolution_t(t, n, array_of_convolutions,x_axis):
    """
    Evaluate the n-fold convolution at a given time t using linear interpolation.

    :param t: time at which to evaluate the convolution
    :param conv_array: output from n_convolution
    :return: interpolated convolution value at time t
    """
    evaluation_array = array_of_convolutions[n-1][:]

    if t <= x_axis[0]:
        return evaluation_array[0]
    elif t >= x_axis[-1]:
        return evaluation_array[-1]
    else:
        return np.interp(t, x_axis, evaluation_array)

if __name__=="__main__":
    alpha= 1
    beta = 2.5
    T=5

    absc= np.linspace(0,beta*T)
    import matplotlib.pyplot as plt
    plt.plot(absc, cosinusoidal_kernel(absc, alpha,beta))
    plt.plot(absc, sinusoidal_kernel(absc, alpha, beta))

    plt.show()

def is_exponential_decay(func):
    """
    Verify if `func` is of the form:
        lambda x: exponential_kernel(x, alpha=..., beta=...)
    with arbitrary alpha and beta values.
    """
    # Must be a lambda function
    if not isinstance(func, types.LambdaType) or func.__name__ != "<lambda>":
        return False

    # Disassemble the function bytecode
    instructions = list(dis.get_instructions(func))

    # Look for a call to "example" with keywords alpha and beta
    seen_call = False

    for instr in instructions:
        if (instr.opname == "LOAD_GLOBAL" or instr.opname =="LOAD_ATTR") and instr.argval == "exponential_kernel":
            seen_call = True

    return seen_call


class ExponentialDecay():
    def __init__(self,alpha,beta):
        self.func = lambda x: exponential_kernel(x, alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta
        self.is_exponential_decay = True

