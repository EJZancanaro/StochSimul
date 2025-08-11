import new_model
import basicfunctions
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_POINTS_PER_UNIT_OF_TIME = 10**3

class NeuronA():
    print("The class SimpleNeuronSim.NeuronA is an old, less flexible version of MultipleNeuronSim.NeuronLinear.")
    def __init__(self,phi,mu,T,M, poisson_measure):
        self.mu = mu
        self.phi = phi
        self.T = T
        self.M = M
        self.poisson_measure=poisson_measure

        self.history=self.events()
    def events(self):
        return new_model.simulate_hawkes_linear_finite2D(mu=self.mu,phi=self.phi,poisson_measure=self.poisson_measure)

    def intensity_values(self):
        time_scale = np.linspace(0,self.T,T*NUMBER_OF_POINTS_PER_UNIT_OF_TIME)
        intensity_values = np.zeros_like(time_scale)

        #for (i,t) in enumerate(time_scale):
         #   intensity_values[i] = new_model.hawkes_intensity(t = t,history = self.history, mu=self.mu,phi=self.phi)
        intensity_values = new_model.hawkes_intensity_array(time_scale_array=time_scale,
                                                            history=self.history, mu=self.mu, phi=self.phi)
        return intensity_values,time_scale
    def intensity_plot(self,string):
        intensity_values, time_scale = self.intensity_values()
        plt.plot(time_scale,intensity_values,label=string)

class NeuronB():
    print("The class SimpleNeuronSim.NeuronB is an old, less flexible version of MultipleNeuronSim.NeuronSemilinear.")
    def __init__(self,phiB,muB,T,M, neuronA,poisson_measure):
        self.phiB = phiB
        self.muB = muB
        self.T = T
        self.M = M
        self.neuronA = neuronA
        self.poisson_measure = poisson_measure

        self.history = self.events()
    def events(self):

        return new_model.simulate_hawkes_semilinear_finite2D(muB=self.muB, muA=self.neuronA.mu, phiA= self.neuronA.phi,
                                                             phiB=self.phiB,poisson_measure=self.poisson_measure,
                                                             eventsA = self.neuronA.history)
    def intensity_values(self):
        time_scale = np.linspace(0,self.T,T*NUMBER_OF_POINTS_PER_UNIT_OF_TIME)

        #for (i,t) in enumerate(time_scale):
         #   intensity_values[i] = new_model.hawkes_intensity(t = t,history = self.history, mu=self.muB,phi=self.phiB)
        intensity_values = new_model.hawkes_intensity_array(time_scale_array=time_scale,
                                                            history=self.history,mu=self.muB,phi=self.phiB)
        return intensity_values,time_scale
    def intensity_plot(self,string):
        intensity_values, time_scale = self.intensity_values()
        plt.plot(time_scale,intensity_values,label=string)


if __name__ == "__main__":
    T = 30
    M = 1000

    muA = 2
    muB = 4
    phiA = lambda x: basicfunctions.exponential_kernel(x,alpha=3.5,beta=4)
    phiB = lambda x: basicfunctions.exponential_kernel(x,alpha=4,beta=3.8)

    poisson_measure_A = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measure_B = new_model.generate_Poisson_2D_finitearea(T=T,M=M)

    neuronA = NeuronA(mu=muA,phi=phiA, poisson_measure=poisson_measure_A,T=T,M=M)
    neuronB = NeuronB(muB=muB,phiB=phiB, poisson_measure=poisson_measure_B,neuronA=neuronA,T=T,M=M)
    linear_neuronB = NeuronA(mu=muB,phi=phiB,poisson_measure=poisson_measure_B,T=T,M=M)
    plt.figure(0)
    neuronA.intensity_plot("Intensity of neuron A")
    neuronB.intensity_plot("Intensity of neuron B")
    linear_neuronB.intensity_plot("Intensity of neuron B uninhibited by A")
    plt.legend()
    plt.show()
