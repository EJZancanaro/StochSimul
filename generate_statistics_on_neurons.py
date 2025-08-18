import MultipleNeuronSim
import new_model
import basicfunctions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__=="__main__" :
    T = 100
    M = 1000

    MAX_MU_A = 100
    MAX_MU_B = 100

    MAX_ALPHA_A = 100
    MAX_ALPHA_B = 100

    MAX_BETA_A = 100
    MAX_BETA_B = 100

    N_SIMULATIONS_MAX = 1000

    with open("results.txt","w") as file :
        file.write(f"Simulations done with T={T} M={M} MAX_MU_A={MAX_MU_A} MAX_MU_B={MAX_MU_B} MAX_ALPHA_A={MAX_ALPHA_A} MAX_ALPHA_B={MAX_ALPHA_B} MAX_BETA_A{MAX_BETA_A} MAX_BETA_B{MAX_BETA_B}\n")
        file.write("muA,muB,alphaA,alphaB,betaA,betaB,explodedA,explodedB\n")

        for _ in tqdm(range(N_SIMULATIONS_MAX)) :

            poisson_measureA1 = new_model.generate_Poisson_2D_finitearea(T=T, M=M)
            poisson_measureB1 = new_model.generate_Poisson_2D_finitearea(T=T, M=M)

            muA = MAX_MU_A*np.random.rand()
            muB = MAX_MU_B*np.random.rand()

            alpha_A = MAX_ALPHA_A*np.random.rand()
            alpha_B = MAX_ALPHA_B*np.random.rand()

            beta_A = MAX_BETA_A*np.random.rand()
            beta_B = MAX_BETA_B*np.random.rand()


            phiA = lambda x: basicfunctions.exponential_kernel(x, alpha=alpha_A, beta=beta_A)


            phiB11 = lambda x: basicfunctions.exponential_kernel(x, alpha=alpha_B, beta=beta_B)

            A1 = MultipleNeuronSim.NeuronLinear(initial_intensity=muA, kernel_function=phiA, poisson_measure=poisson_measureA1)

            parent_kernels_B1 = [phiB11]

            B1 = MultipleNeuronSim.NeuronSemilinear(initial_intensity=muB, list_parent_kernels=[phiB11],
                                  poisson_measure=poisson_measureB1)

            MultipleNeuronSim.NeuronSemilinear.events()

            array_valuesA, trash = A1.intensity_values()

            array_valuesB, trash = B1.intensity_values()

            # given the limitations of a bounded M, we will suppose that if A exploded we are not really interested in the observation

            explodedA = ( max(array_valuesA) > M )
            explodedB = ( max(array_valuesB) > M )

            file.write(f"{muA},{muB},{alpha_A},{alpha_B},{beta_A},{beta_B},{explodedA},{explodedB}\n")

            MultipleNeuronSim.NeuronSemilinear.reinitialise()
            MultipleNeuronSim.NeuronLinear.reinitialise()



