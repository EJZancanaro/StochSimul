import basicfunctions
import basicprocesses
import numpy as np
import matplotlib.pyplot as plt

def affiche_points(array): #Affiche les (Tk) sur l'axe des abscisses lorsqu'ils sont donnés
    plt.plot(array,np.zeros( len(array)),".")

affiche_points(np.array([0,3,5]))

def intensity_test(t):
    return 1 + np.sin(t)


basicprocesses.inhomogeneous_poisson(intensity_test,T=5, _explanatoryplot=True)

plt.show()