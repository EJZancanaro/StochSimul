import new_model
import new_model_fast
import numpy as np
import basicfunctions
import time

def test_hawkes_intensity_fast():
    T=20
    M=2000
    time_scale = np.linspace(0,T, T*10**3)

    #Try three cases : almost constant, non-explosive, explosive

    mus  =   [1, 7  , 13]
    alphas = [3, 3  , 20]
    betas =  [8, 3.5, 1]

    for i in range(3):
        mu = mus[i]
        alpha = alphas[i]
        beta = betas[i]
        phi = lambda x: basicfunctions.exponential_kernel(x,alpha = alpha, beta=beta)

        poisson_measure = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
        history = new_model.simulate_hawkes_linear_finite2D(mu=mu,phi=phi,poisson_measure=poisson_measure)
        start = time.perf_counter()

        intensity_slow = None
        for t in time_scale :
            intensity_slow = new_model.hawkes_intensity(t,history[t>history], mu, phi)
        end = time.perf_counter()

        time_slow_version = end-start

        start = time.perf_counter()
        intensity_fast=None
        previous_t = None
        for t in time_scale:
            intensity_fast = new_model_fast.hawkes_intensity_fast(t, history[history<t], mu, phi,
                                                                  previous_value_intensity=intensity_fast,
                                                                  previous_time=previous_t,
                                                                  is_exponential_decay=True)
            previous_t = t
        end = time.perf_counter()

        time_fast_version = end-start

        start = time.perf_counter()
        intensity_array = new_model.hawkes_intensity_array(time_scale_array=time_scale,
                                                           history=history,mu=mu,phi=phi)
        end = time.perf_counter()
        time_array_version = end-start

        start = time.perf_counter()

        intensity_array_fast = new_model_fast.hawkes_intensity_fast_array(time_scale_array=time_scale,history=history, mu=mu, phi=phi,
                                                                          is_exponential_decay=True)
        end = time.perf_counter()
        time_array_fast_version = end-start
        print("------------------")
        print(f"slow intensity: mu={mu}, alpha={alpha},beta={beta}", intensity_slow)
        print(f"fast intensity: mu={mu}, alpha={alpha},beta={beta} ", intensity_fast)
        print(f"array based intensity: mu={mu}, alpha={alpha},beta={beta} ",intensity_array[-1])
        print(f"fast array based intensity: mu={mu}, alpha={alpha},beta={beta} ",intensity_array_fast[-1])
        print("\n")
        print(f"time fast method: mu={mu}, alpha={alpha},beta={beta} ", time_fast_version)
        print(f"time slow method: mu={mu}, alpha={alpha},beta={beta} ", time_slow_version)
        print(f"array based method: mu={mu}, alpha={alpha},beta={beta} ", time_array_version)
        print(f"fast array based method: mu={mu}, alpha={alpha},beta={beta} ", time_array_fast_version)
        print("------------------")
        assert np.isclose(intensity_slow, intensity_fast) #tests if the functions output the same

        assert np.isclose(intensity_array[-1], intensity_fast)

        assert np.isclose(intensity_array_fast[-1], intensity_fast)

        if len(history)> 3 :
            assert time_slow_version>time_fast_version

            assert time_array_version>time_fast_version

            assert time_array_version>time_array_fast_version

            assert time_fast_version > time_array_fast_version
            #when there's enough events in the history, tests if slow truly is slower
            #when there are less than 2 events, numpy's speed makes it so it isn't worth it to use the "fast method"


def test_simulate_linear_Hawkes_fast():
    T = 10
    M = 1000

    poisson_measure = new_model.generate_Poisson_2D_finitearea(T=T,M=M)

    mu = 3
    alpha = 5
    beta = 6

    phi = lambda x : basicfunctions.exponential_kernel(x, alpha = alpha, beta = beta)

    start = time.perf_counter()
    slow_hawkes = new_model.simulate_hawkes_linear_finite2D(mu=mu, phi=phi, poisson_measure=poisson_measure)
    end = time.perf_counter()

    slow_time = end - start

    start = time.perf_counter()
    fast_hawkes = new_model_fast.simulate_hawkes_linear_finite2D_fast(mu= mu, phi=phi,
                                                                      poisson_measure=poisson_measure,
                                                                      is_exponential_decay=True)
    end = time.perf_counter()

    fast_time = end-start


    assert len(slow_hawkes)==len(fast_hawkes)

    assert np.all(np.isclose(slow_hawkes, fast_hawkes) )

    if len(fast_hawkes)>3 :
        print("slow hawkes",slow_time)
        print("fast hawkes",fast_time)
        assert slow_time > fast_time

def test_simulate_semilinear_Hawkes_fast():
    T = 30
    M = 1000

    poisson_measureA = new_model.generate_Poisson_2D_finitearea(T=T, M=M)
    poisson_measureB = new_model.generate_Poisson_2D_finitearea(T=T, M=M)

    muA = 3
    muB = 12
    alphaA = 3
    alphaB = 9
    betaA = 4
    betaB = 3
    phiA = lambda x: basicfunctions.exponential_kernel(x, alpha=alphaA, beta=betaA)
    phiB = lambda x : basicfunctions.exponential_kernel(x,alpha = alphaB, beta= betaB)

    eventsA = new_model.simulate_hawkes_linear_finite2D(mu=muA, phi=phiA,poisson_measure=poisson_measureA)

    start = time.perf_counter()
    slow_hawkes = new_model.simulate_hawkes_semilinear_finite2D(muB=muB, phiB=phiB, muA=muA, phiA=phiA,
                                                                eventsA=eventsA, poisson_measure=poisson_measureB)
    end = time.perf_counter()

    slow_time = end - start

    start = time.perf_counter()
    fast_hawkes = new_model_fast.simulate_hawkes_semilinear_finite2D_fast(muB=muB, phiB=phiB, muA=muA, phiA=phiA,
                                                                eventsA=eventsA, poisson_measure=poisson_measureB,
                                                                          is_exponential_decay=True)
    end = time.perf_counter()

    fast_time = end - start

    assert len(slow_hawkes) == len(fast_hawkes)

    assert np.all(np.isclose(slow_hawkes, fast_hawkes))

    if len(slow_hawkes) > 2:
        print("slow hawkes", slow_time)
        print("fast hawkes", fast_time)
        assert slow_time > fast_time