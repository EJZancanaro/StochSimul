
import numpy as np
import basicfunctions

def generate_Poisson_2D_finitearea(T, M):
    """
    Generates a poisson realisation in the 2D space [0,T] x [0,M]
    :param T: Time bound
    :param M: Intensity bound. Should approach infinity if we desire to simulate Hawkes processes with it.
    :return: A dictionary containing all the parameters of the poisson simulation as well as the poisson realisation points' 2D coordinates
    """
    assert T>0
    assert M>0

    number_poisson_results = np.random.poisson(T * M)

    array_poisson_realisations = T * np.random.rand(number_poisson_results)
    array_poisson_realisations = np.sort(array_poisson_realisations)

    array_theta = M * np.random.rand(number_poisson_results)

    return_dict = {
        "time" : array_poisson_realisations,
        "theta": array_theta,
        "T":T,
        "M":M
    }
    return return_dict


def hawkes_intensity(t, history, mu, phi):
    """Compute intensity at time t given history and kernel phi.
    t: time of evaluation
    history : array of events of the process. Must be a numpy array
    mu: initial intensity
    phi: kernel of the increments of the process.
    """

    #Attention. If a function calls this function inside a loop where history
    #is filled progressively, then that loop will run on O(len(final_history)**2) instead of
    #O(len(final_history). This should be adressed in a following update
    history = np.asarray(history)
    if history.size == 0:
        return mu
    else:
        deltas = t - history[history < t]

        return mu + np.sum(phi(deltas))


def hawkes_intensity_array(time_scale_array,history,mu,phi):
    """
    Computes the hawkes intensity over a time interval
    :param time_scale_array: time interval
    :param history: all events up to the end of the interval of the process
    :param mu: initial intensity of the hawkes process
    :param phi: kernel function of the hawkes process
    :return: array of the intensities throughout the timescale
    """
    if history.size == 0:
        return mu*np.ones_like(time_scale_array)
    else:
        function = lambda t : mu+np.sum(phi(t-history[history<t]))
        return np.array(list(map(function,time_scale_array))) #quirk of the map function, we have to convert its return
                                                            #to a list before converting it to a numpy array

def simulate_hawkes_linear_finite2D(mu, phi, poisson_measure):
    """
    Simulate a linear Hawkes process by thinning of a poisson measure restricted to a rectangle of R2.

    Parameters:
    - mu: Baseline intensity
    - phi: Function phi(t), the kernel
    - poisson_measure : returned by, generatePoisson2D result of a poisson simulation on a [0,T] x [O,M] grid

    """
    times, thetas = poisson_measure["time"], poisson_measure["theta"]
    hawkes_event = []

    for t, h in zip(times, thetas):

        lambda_t = hawkes_intensity(t, np.array(hawkes_event), mu, phi)
        if h <= lambda_t:
            hawkes_event.append(t)

    return np.array(hawkes_event)


def simulate_hawkes_semilinear_finite2D(muB, phiB, muA, phiA ,eventsA ,poisson_measure):
    times, thetas = poisson_measure["time"], poisson_measure["theta"]
    eventsB = []

    for t, h in zip(times, thetas):
        lambdaB_t = hawkes_intensity(t, np.array(eventsB), muB, phiB)
        lambdaA_t = hawkes_intensity(t, np.array(eventsA), muA, phiA)
        if h < max(0,lambdaB_t-lambdaA_t):
            eventsB.append(t)

    return np.array(eventsB)


