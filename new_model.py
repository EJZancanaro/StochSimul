
import numpy as np
from numpy.ma.core import argmax


def generate_Poisson_2D_finitearea(T, M):
    number_poisson_results = np.random.poisson(T * M)

    array_poisson_realisations = np.random.rand(number_poisson_results)
    array_poisson_realisations = np.sort(array_poisson_realisations)

    array_theta = M * np.random.rand(number_poisson_results)

    return_dict = {
        "time" : array_poisson_realisations,
        "theta": array_theta
    }
    return return_dict

def increment_intensity(t,last_value,phi,past_events):

    return last_value + phi(t-past_events[-1])



def intensity_Hawkes_2Dfinite(T,M, muB, phi,poisson_measure, intensity_external=None):
    if intensity_external is not None:
        muA=intensity_external[0]



    poisson_times=np.sort(poisson_measure["time"])
    poisson_thetas=poisson_measure["theta"]

    nb_time_discretisation = T*10**3

    time_scale = np.linspace(0, T, nb_time_discretisation)

    poisson_times_index = np.array(list(map( lambda x : int(x), poisson_measure["time"]*nb_time_discretisation/T ) ) )
    poisson_thetas_index= np.array(list(map( lambda x : int(x), poisson_measure["theta"]*nb_time_discretisation/T ) ) )

    intensity_values = np.zeros(nb_time_discretisation)
    intensity_values[0:poisson_times[0]] = muB
    """
    ####ne fonctionne pas
    for i in range(1,len(poisson_times)-1): #the case i = 0 is already set at 0

        intensity_values[poisson_times_index[i]:poisson_times_index[i+1]] += [
            ( poisson_thetas_index[i]<=intensity_values[poisson_times_index[i]] )
            for t in np.arange(poisson_times_index[i],poisson_times_index[i+1])
            ]"""

    #criterium_function should be able to be this
    for (last_poisson_index, last_time_poisson) in enumerate(poisson_times):
        next_time_poisson = poisson_times[last_poisson_index+1]

        if poisson_thetas[last_poisson_index]< intensity_values[-1]:
            time_scale_index_lastevent = (time_scale[time_scale>last_time_poisson])[0]
            time_scale_index_nextevent = (time_scale[time_scale<last_time_poisson])[-1]

            intensity_values[time_scale_lastevent:time_scale_nextevent] = intensity_values[-1]+phi(time_scale[max])
    """
    if intensity_external is None : #linear model
        for i,t in enumerate(np.linspace(0,T,nb_time_discretisation)):
            times_mask = (poisson_times<t)

            last_incident_index= np.argmax(poisson_times[poisson_times<t]) if poisson_times[poisson_times<t].size!=0  else -1

            print("index of last poisson",last_incident_index)

            if last_incident_index ==-1:
                intensity_values[i + 1] = intensity_values[i]
            else:
                intensity_values[i+1]=  intensity_values[i] + (
                        phi(t-poisson_times[last_incident_index]) * (poisson_thetas[i] <= intensity_values[i] )
                )


    else :
        for t in np.linspace(0,T,nb_time_discretisation):
            times_mask = (poisson_times<t)
            intensity_mask = (intensity_external<t)

            intensity_values=sum(phi(t-poisson_times[times_mask]) * (
                    poisson_thetas[times_mask] <= intensity_external[intensity_mask] )
                                 )
    """
    ##############

def Hawkes_finite2D(T,M, intensity_array,poisson_measure):
    nb_time_discretisation = len(intensity_array) #Symbolizes the number of floats
                                                # in the floating point arithmetic
                                                #that we will use to model the segment [0,T]
    assert nb_time_discretisation > 10*T

    Hawkes_counting = np.zeros(nb_time_discretisation)

    poisson_times = poisson_measure["time"]
    poisson_thetas = poisson_measure["theta"]

    #######Hawkes jumping times
    Hawkes_events = []
    for i in range(1,len(poisson_times)-1):
        if poisson_thetas[i] <= intensity_array[int(poisson_times[i] * nb_time_discretisation/T) ] :
            Hawkes_events.append(poisson_times[i])
    return np.array(Hawkes_events)
"""
def generate_semilinear_model(T, M, muA, muB, phiBB,phiAA,):

    lambdaA = muA+integral()
    HA =
"""
T=5
M=3

poisson = generate_Poisson_2D_finitearea(T,M)
H = Hawkes_finite2D(T=5, M=3, intensity_array=3*np.ones(5*10**3), poisson_measure=poisson)
#print(H)

#print(poisson["time"].size)
#print(H.size)

mu = 3
alpha = 1
beta = 2
phi = lambda s: alpha * np.exp(-beta * s)

intensity_Hawkes_2Dfinite(T,M,mu,phi, poisson_measure=poisson)