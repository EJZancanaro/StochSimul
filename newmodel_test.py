import numpy as np
import scipy as sp
import new_model
import basicfunctions
from basicfunctions import n_convolution
from new_model import generate_Poisson_2D_finitearea


def test_poisson_2D():

    new_model.generate_Poisson_2D_finitearea(T=4,M=7)
    new_model.generate_Poisson_2D_finitearea(T=1, M=10**6)
    new_model.generate_Poisson_2D_finitearea(T=10**6, M=13)

def test_poisson_is_correct():
    """
    tests if the empirical mean of multiple identically distributed
    poisson simulations converge to their mean at the speed given by the CLT
    """
    s1,s2,s3 = 0,0,0
    N = 10**4
    T1, M1 = 4 , 7
    T2, M2 = 10**3, 13
    T3, M3 = 2, 10**3

    for iteration in range(1,N):
        s1 += ( new_model.generate_Poisson_2D_finitearea(T=T1,M=M1) )["time"].size
        s2 += ( new_model.generate_Poisson_2D_finitearea(T=T2,M=M2) )["time"].size
        s3 += ( new_model.generate_Poisson_2D_finitearea(T=T3,M=M3) )["time"].size
    assert np.abs(s1/N - T1*M1)<1/np.sqrt(N) * T1*M1
    assert np.abs(s2/N - T2*M2)<1/np.sqrt(N) * T2*M2
    assert np.abs(s3/N -T3*M3) < 1/np.sqrt(N) * T3 * M3

def test_hawkes_intensity():
    new_model.hawkes_intensity(t=5, history=np.array([]),mu=3,phi=lambda x: basicfunctions.exponential_kernel(x, alpha=1,beta=2))
    new_model.hawkes_intensity(t=0, history=np.array([0.27354, 0.438454]), mu=3,
                               phi=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=2))
    poisson_measure = new_model.generate_Poisson_2D_finitearea(T=5, M=20)
    new_model.simulate_hawkes_linear_finite2D(mu=3, phi=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=2),
                                    poisson_measure=poisson_measure)

def test_linear_hawkes_2D():
    tests_T = [3,4,6,23]
    tests_M = [3,4,6,23]
    for t in tests_T:
        for m in tests_M:
            poisson_dict = generate_Poisson_2D_finitearea(T=t,M=m)
            new_model.simulate_hawkes_linear_finite2D(mu=5,
                                                      phi=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=2),
                                                      poisson_measure=poisson_dict)
def test_linear_Hawkes_is_correct():
    T = 10
    M = 1000
    mu = 5
    poisson_dict = generate_Poisson_2D_finitearea(T=T, M=M)
    func = lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=6)
    Hawkes_events = new_model.simulate_hawkes_linear_finite2D(mu=mu,
                                              phi=func,
                                              poisson_measure=poisson_dict)

    resolution = 10 ** 4
    x_axis = np.linspace(0,T, T*resolution )

    simulated_intensity = new_model.hawkes_intensity(t=T, history =Hawkes_events, mu=mu, phi=func)

    n_max=1000
    convs = n_convolution(phi=func, n=n_max, x_axis=x_axis)  # shape: (n, len(x_axis))
    psi = convs.sum(axis=0)  # shape: (len(x_axis),)
    #print(psi.size==len(x_axis)) returns true, as desired

    real_intensity = mu*(1+sp.integrate.trapezoid(y=psi, x=x_axis))

    assert np.abs(real_intensity - simulated_intensity) < 10.**(-1)*real_intensity

def test_semilear_Hawkes():
    new_model.simulate_hawkes_semilinear_finite2D(muB=3,phiB=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=3),
                                                  muA=4,
                                                  eventsA=new_model.simulate_hawkes_linear_finite2D(mu=4,phi=lambda x: basicfunctions.exponential_kernel(x, alpha=3, beta=6),
                                                                                                    poisson_measure=new_model.generate_Poisson_2D_finitearea(T=5,M=20)),

                                                  )