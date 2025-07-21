import numpy as np
import matplotlib.pyplot as plt
import basicprocesses


def plot_Hawkes_expdecay(T, mu, alpha, beta, set_instants):
    #TODO add to documentation what exactly the solution was
    """
    Plots efficiently a Hawkes function with exponential decay by using a local property of this particular intensity function

    :param T: endtime of the process
    :param mu: original and baseline value of the intensity
    :param alpha: increment of the intensity at each time step
    :param beta: exponential decay of the intensity parameter
    :param set_instants: array of time instances of the process
    :return:
    """
    set_instants = np.array(set_instants)
    plt.figure()
    plt.title("Affichage de l'intensité avec une écriture locale")
    abscisse_courante = np.linspace(0, set_instants[0], 10 ** 3)
    taille_abscisse = len(abscisse_courante)

    plt.plot(abscisse_courante, mu * np.ones(taille_abscisse))
    plt.xlim(0, T)
    dernier_t = set_instants[0]

    for t in set_instants[1:]:
        dernier_lambda = basicprocesses.Hawkes_expdecay_intensity(dernier_t, mu, alpha, beta, set_instants[set_instants < dernier_t])

        abscisse_courante = np.linspace(dernier_t, t, 10 ** 3)
        plt.plot(abscisse_courante,
                 (dernier_lambda + alpha - mu) * np.exp(-beta * (abscisse_courante - dernier_t)) + mu)

        dernier_t = t