import basicfunctions
import matplotlib.pyplot as plt
import numpy as np
import specialplottings

def homogenous_poisson(intensity,T):
    """Simulates a homogenous Poisson Process starting at 0 and ending at time T
    params:
        intensity: value of the instensity \lambda
        T: Time of end of simulation
    returns:
        t_array: list of instants of the process
    """
    t_array = [0]
    while True:
        u = np.random.rand()
        w = -np.log(u) / l
        t_array.append(t_array[-1] + w)
        if t_array[-1] > T:
            return t_array[1:-1]


def inhomogeneous_poisson(intensity, T, _explanatoryplot=False):
    # Construit un IPP par méthode du rejet avec une v.A auxiliare HPP
    """
    Constructs an inhomogeneous Poisson Process starting at 0 and ending at time T, by a rejection method based on an homogeneous Poisson process
    :param intensity: function, intensity of the intended Poisson process
    :param T: Time of end of simulation
    :param _explanatoryplot: Plotting of the rejection procedure
    :return: list of instants of the process
    """
    l_sup = basicfunctions.supremum(intensity,0, T)

    s_array = [0]  # liste des resultats d'un HPP (densité auxiliaire pour construire IPP)
    t_array = [0]  # liste des resultats d'un IPP

    if _explanatoryplot:  # Je plot l'intensité
        plt.figure()
        plt.title("Simulation par rejet d'une IPP grâce à une HPP")
        N = 100
        abscisse = np.linspace(0, T, N)
        plt.plot(abscisse, intensity(abscisse) / l_sup, label="Intensité renormalisée")

    while s_array[-1] < T:
        u = np.random.rand()
        w = -np.log(u) / l_sup
        s_array.append(s_array[-1] + w)
        D = np.random.rand()

        if D < intensity(s_array[-1]) / l_sup:
            t_array.append(s_array[-1])
            if _explanatoryplot and s_array[-1] < T:  # Plot du point lorsqu'il est accepté
                plt.plot(s_array[-1], D, ".", color='g')

        elif _explanatoryplot and s_array[-1] < T:  # Plot du point lorsqu'il est rejeté
            plt.plot(s_array[-1], D, "x", color='r')

    if _explanatoryplot:
        plt.legend()

    if t_array[-1] < T:
        return t_array
    else:
        return t_array[:-1]


def Hawkes_expdecay_intensity(s, mu, alpha, beta, set_instants):
    """
    Describes the intensity of a given Hawkes Process with exponential decay

    :param s: point of evaluation
    :param mu: initial value and baseline of the intensity functions
    :param alpha: increment of intensity value at each time of realisation
    :param beta: coefficient of exponential of decay
    :param set_instants: instants of the process
    :return:
    """
    return mu + sum(alpha * np.exp(-beta * (s - np.array(set_instants))))


def HawkesOgataThining(mu, alpha, beta, T,_explanatoryplot=False):
    ###TODO :For now only able to handle the exponential decay Hawkes intensity, fix that
    """
    This uses Ogata thinning alogorithm to obtain the instants of a realisation of a Hawkes process
    :param mu:
    :param alpha:
    :param beta:
    :param T:
    :return:
    """

    set_instants = []
    s = 0
    t_array = []

    while s < T:
        l_bar = Hawkes_expdecay_intensity(s, mu, alpha, beta, set_instants)
        u = np.random.rand()
        w = -np.log(u) / l_bar
        s = s + w

        D = np.random.rand()

        if D * l_bar <= Hawkes_expdecay_intensity(s, mu, alpha, beta, set_instants):
            t_array.append(s)
            set_instants.append(t_array[-1])

    if t_array[-1] <= T:

        return t_array
    else:
        if _explanatoryplot:
            #TODO again, the following only works for exponential decay Hawkes processes
            specialplottings.plot_Hawkes_expdecay(t_array[:-1], T, mu, alpha, beta, set_instants[:-1])
            # Ils ne sont pas tout à fait égaux pour des betas trop grand : instabilité ?

        return t_array[:-1]
