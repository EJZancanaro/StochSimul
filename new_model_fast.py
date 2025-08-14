import types

import numpy as np
import new_model
import basicfunctions


def hawkes_intensity_fast(t, history,mu,phi,previous_value_intensity=None,previous_time = None,is_exponential_decay=False):
    """
    Compute intensity at time t given history and kernel phi.
    Is faster than hawkes_intensity when computing successive values of intensity in a loop if phi is an exponential decay function.

    :param t: Time of evaluation
    :param history: list or array of events anterior to t
    :param mu: initial intensity
    :param phi: kernel of the increments of the process.
    :param previous_value_intensity:previous output of the intensity
    :return: intensity of the corresponding process at time t
    """

    params = [cell.cell_contents for cell in phi.__closure__]
    alpha = params[0]
    beta = params[1]

    if is_exponential_decay:
        if previous_value_intensity is None:
            # No information about previous intensities, compute normally
            return new_model.hawkes_intensity(t,history,mu,phi)
        else:
            #assert previous_time is not None

            intensity = mu + np.exp(-beta*(t-previous_time))*(previous_value_intensity - mu)+ alpha*sum(np.exp(-beta*(t-history[(history<t) & (history>previous_time) ])))
            return intensity

    else:
        # if it's not exponential decay, we can't be faster without
        # either having more information on the function or making taylor approximations
        return new_model.hawkes_intensity(t,history,mu,phi)

def hawkes_intensity_fast_array(time_scale_array, history,mu, phi,is_exponential_decay=False):
    intensity_array = np.zeros_like(time_scale_array)
    previous_value_intensity = None
    previous_time=None
    if not is_exponential_decay :
        return new_model.hawkes_intensity_array(time_scale_array=time_scale_array,history=history ,mu=mu, phi=phi)
    else:
        for (i,t) in enumerate(time_scale_array) :
            previous_value_intensity = hawkes_intensity_fast(t=t, history=history, mu=mu, phi=phi,
                                    previous_value_intensity=previous_value_intensity, previous_time=previous_time
                                   ,is_exponential_decay=is_exponential_decay)
            intensity_array[i] = previous_value_intensity
            previous_time = t
        return intensity_array

def simulate_Hawkes_linear_finite2D_fast(mu, phi , poisson_measure,is_exponential_decay=False):
    if not is_exponential_decay :
        return new_model.simulate_hawkes_linear_finite2D(mu,phi,poisson_measure)

    times, thetas = poisson_measure["time"], poisson_measure["theta"]
    hawkes_event = []

    lambda_t_right = None
    lambda_t_left = None
    previous_t = None
    for t, h in zip(times, thetas):

        lambda_t_left = hawkes_intensity_fast(t=t, history=np.array(hawkes_event),
                                         mu=mu, phi=phi,previous_value_intensity=lambda_t_right,
                                         previous_time=previous_t,
                                         is_exponential_decay=is_exponential_decay)
        previous_t = t
        if h <= lambda_t_left:
            hawkes_event.append(t)
            lambda_t_right = lambda_t_left + phi(0)
        else:
            lambda_t_right = lambda_t_left
    #Remark :  we need to include this left/right logic not present in the slow version


    return np.array(hawkes_event)
