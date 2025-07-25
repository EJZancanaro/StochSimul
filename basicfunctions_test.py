import numpy as np

import basicfunctions

def test_n_convolution():
    x_axis = np.linspace(0,5,5*10**3)
    basicfunctions.n_convolution(phi=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=2),
                                 n=4,
                                 x_axis=x_axis,)

def test_n_convolution_is_correct():
    M = 5
    n=2
    x_axis = np.linspace(0,M,M*5*10**3)
    all_convs_simulated = basicfunctions.n_convolution(phi=lambda x: np.exp(x), n=2,x_axis=x_axis)

    for t in np.arange(M/100,M,M/100):
        simulated_t = basicfunctions.n_convolution_t(t,n=2, array_of_convolutions=all_convs_simulated,x_axis=x_axis)
        real_t = t*np.exp(t)

        assert np.abs(simulated_t - real_t)< 10.**(-2)*real_t #The relative error must be of at most one significant digit

