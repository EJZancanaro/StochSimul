import numpy as np

import basicfunctions

def test_n_convolution():
    basicfunctions.n_convolution(phi=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=2),
                                 n=4, start_interval=0, end_interval=5)

def test_n_convolution_is_correct():
    M = 5
    simulated = basicfunctions.n_convolution(phi=lambda x: np.exp(x), n=2,start_interval=0, end_interval=M)
    for t in np.arange(M/100,M,M/100):
        simulated_t = basicfunctions.n_convolution_t(t, simulated)
        real_t = t*np.exp(t)

        assert np.abs(simulated_t - real_t)< 10.**(-1)*real_t #The relative error must be of at most one significant digit

