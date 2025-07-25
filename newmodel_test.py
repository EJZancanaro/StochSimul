import numpy as np
import scipy as sp
import new_model
import basicfunctions
from basicfunctions import n_convolution
from new_model import generate_Poisson_2D_finitearea
from scipy.stats import kstest

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
        poisson1= ( new_model.generate_Poisson_2D_finitearea(T=T1,M=M1) )
        poisson2= ( new_model.generate_Poisson_2D_finitearea(T=T2,M=M2) )
        poisson3= ( new_model.generate_Poisson_2D_finitearea(T=T3,M=M3) )
        s1 += poisson1["time"].size
        s2 += poisson2["time"].size
        s3 += poisson3["time"].size

    # Tests if the average number of points is the expected
    assert np.abs(s1/N - T1*M1)<1/np.sqrt(N) * T1*M1
    assert np.abs(s2/N - T2*M2)<1/np.sqrt(N) * T2*M2
    assert np.abs(s3/N -T3*M3) <1/np.sqrt(N) * T3*M3

    #tests if the times generated follow a distribution
    assert kstest(poisson1["time"] / T1, 'uniform').pvalue > 0.01
    assert kstest(poisson2["time"] / T2,'uniform').pvalue > 0.01
    assert kstest(poisson3["time"] / T3,'uniform').pvalue > 0.01
    #tests if the thetas generated follow a uniform distribution

    assert kstest(poisson1["theta"] / M1, 'uniform').pvalue > 0.01
    assert kstest(poisson2["theta"] / M2, 'uniform').pvalue > 0.01
    assert kstest(poisson3["theta"] / M3, 'uniform').pvalue > 0.01


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
    T = 7
    M = 500
    mu = 2

    resolution = 10 ** 3
    x_axis = np.linspace(0, T, T * resolution)

    func = lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=6)

    empirical_average=0
    number_iteration=10**3
    for _ in range(number_iteration):
        poisson_dict = generate_Poisson_2D_finitearea(T=T, M=M)

        Hawkes_events = new_model.simulate_hawkes_linear_finite2D(mu=mu,
                                                  phi=func,
                                                  poisson_measure=poisson_dict)

        #print(Hawkes_events)
        #print(Hawkes_events.size)
        #print("ATTENTION")
        simulated_intensity = new_model.hawkes_intensity(t=T, history=Hawkes_events, mu=mu, phi=func)
        print(simulated_intensity)
        #print("End of ATTENTION")

        empirical_average+=simulated_intensity

    empirical_average /= number_iteration

    max_convolution_n=100
    convs = n_convolution(phi=func, n=max_convolution_n, x_axis=x_axis)  # shape: (n, len(x_axis))
    psi = convs.sum(axis=0)  # shape: (len(x_axis),)
    #print(psi.size==len(x_axis)) returns true, as desired

    real_expected_intensity = mu*(1+sp.integrate.trapezoid(y=psi, x=x_axis))


    assert np.abs(real_expected_intensity - empirical_average) < 1/np.sqrt(number_iteration)*real_expected_intensity





def test_semilinear_Hawkes():
    T = 10
    M=600

    muA = 3
    muB = 2
    poisson_measure_A = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measure_B = new_model.generate_Poisson_2D_finitearea(T=T,M=M)

    new_model.simulate_hawkes_semilinear_finite2D(muB=muB,phiB=lambda x: basicfunctions.exponential_kernel(x, alpha=1, beta=3),
                                                  muA=4,
                                                  eventsA=new_model.simulate_hawkes_linear_finite2D(mu=muA,
                                                                                                    phi=lambda x: basicfunctions.exponential_kernel(x, alpha=3, beta=6),
                                                                                                    poisson_measure=poisson_measure_A),
                                                  poisson_measure=poisson_measure_B
                                                  )
