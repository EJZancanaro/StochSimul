import basicfunctions
import basicprocesses
import numpy as np
import matplotlib.pyplot as plt
import new_model
from projet_stage.StochSimul.basicprocesses import Hawkes_expdecay_intensity
from projet_stage.StochSimul.new_model import integral_wr_poisson_measure, generate_Poisson_2D_finitearea


def affiche_points(array): #Affiche les (Tk) sur l'axe des abscisses lorsqu'ils sont donnés
    plt.plot(array,np.zeros( len(array)),".")

affiche_points(np.array([0,3,5]))

def intensity_test(t):
    return 1 + np.sin(t)

error_counter=0
####basicprocesses.py test
try :
    basicprocesses.inhomogeneous_poisson(intensity_test,T=5, _explanatoryplot=False)

except:
    print("Error in testing basicprocesses.inhomogeneous_poisson")
    error_counter+=1

####new_model.py tests

#try:
T = 5
M=10
mu = 2
alpha=2
beta = 4
poisson_measure = generate_Poisson_2D_finitearea(T,M)
def exponential_decay(x, alpha, beta):
    return alpha*np.exp(-beta*x)
integral_wr_poisson_measure(T,M, poisson_measure, exponential_decay, alpha, beta)


print("Error in testing integral_wr_poisson_measure")
error_counter+=1

if error_counter==0 :
    print("All tests passed")
else:
    exit(1)