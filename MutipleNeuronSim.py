import new_model
import basicfunctions
import numpy as np
import matplotlib.pyplot as plt

from projet_stage.StochSimul.new_model import generate_Poisson_2D_finitearea

NUMBER_OF_POINTS_PER_UNIT_OF_TIME = 10**3


class NeuronLinear():
    number_instances_linear = 0
    list_linear_neurons = []
    def __init__(self, intial_intensity , kernel_function, poisson_measure,T,M):
        self.id = NeuronLinear.number_instances_linear
        NeuronLinear.number_instances_linear+=1
        NeuronLinear.list_linear_neurons.append(self.id)

        self.initial_intensity = intial_intensity
        self.kernel_function = kernel_function
        self.poisson_measure = poisson_measure
        self.T = T
        self.M = M

        self.history = self.events()

    def events(self):
        return new_model.simulate_hawkes_linear_finite2D(mu=self.initial_intensity, phi=self.kernel_function,
                                                         poisson_measure=self.poisson_measure)
    def intensity_values(self):
        time_scale = np.linspace(0, self.T, T * NUMBER_OF_POINTS_PER_UNIT_OF_TIME)

        intensity_values = new_model.hawkes_intensity_array(time_scale_array=time_scale,
                                                            history=self.history, mu=self.initial_intensity,
                                                            phi=self.kernel_function)
        return intensity_values, time_scale

    def intensity_plot(self, string):
        intensity_values, time_scale = self.intensity_values()
        plt.plot(time_scale, intensity_values, label=string)


class NeuronSemilinear() :
    number_instances_semilinear = 0
    list_semilinear_neurons = []
    def __init__(self,initial_intensity,T,M):
        assert NeuronLinear.list_linear_neurons != []
        #It is imperative all linear Neurons are defined before any semilinear neurons

        self.id = NeuronSemilinear.number_instances_semilinear #integer that identifies the neuron
        NeuronSemilinear.number_instances_semilinear+=1
        NeuronSemilinear.list_semilinear_neurons.append(self)

        self.initial_intensity = initial_intensity
        self.T = T
        self.M = M

        #We must not generate the events before creating all semilinear neurons

    def inform(self,array_influence_functions,array_inhibiting, array_exciting, array_poisson_measure_exciting):
        #Fills in the previously unfilled attributes of the object

        self.array_influence_functions = np.copy(array_influence_functions)
        self.array_exciting = np.copy(array_exciting)
        self.array_inhibiting = np.copy(array_inhibiting)
        self.array_poisson_measure_exciting = np.copy(array_poisson_measure_exciting)

    def is_ready(self):
        assert self.array_influence_functions is not None
        assert self.array_exciting is not None
        assert self.array_inhibiting is not None
        assert self.array_poisson_measure_exciting is not None

    def events(self):
        sum_intensities = 0
        for linear_neuron in NeuronLinear.list_linear_neurons:
            total_events = linear_neuron.history
            #TODO notice that it is not necessary to compute the intensity,
            #we will call the function use to generate semilienar Hawkes processes

            sum_intensities += linear_neuron.intensity_values()


        mean_of_linear_intensities = 1/NeuronLinear.number_instances_linear * sum_intensities



    def intensity_value(self,t,history_of_self,array_influence_functions, array_exciting, array_inhibiting, array_poisson_measures_exciting):
        #Same as the simple case, we suppose we know all events up to time t-, what is the intensity at time t?

        sum_of_integrals = 0

        history=[]
        for neuron in array_exciting:
            integral = 0
            if neuron.history.size==0:
                integral += neuron.initial_intensity
            else :
                integral += neuron.initial_intensity + np.sum(neuron.kernel_function(neuron.history[neuron.history<t]) )

            sum_of_integrals += integral

        Nb_exciting_neurons = NeuronSemilinear.number_instances_semilinear

        assert False #TODO this implementation is false. The list of all kernel functions has to be a matrix.
                     # The current state of the implementation considers that neurons interact with others
                     # with a kernel that is Identical to the one he uses to interact with itself.
                     # This is not true in the general case.

                     #TODO How to solve it : instead of kernel_function as an attribute, every instance
                     # must have as an attribute a  list of kernels, that MUST be the same size as
                     # the list of all existing semilinear neurons.
                     # to acces his own kernel, a neuron need only to acces that list at the index of his own id.
        return self.initial_intensity + 1/Nb_exciting_neurons * sum_of_integrals



if __name__ == "__main__":
    T = 30
    M = 1000
    poisson_measure = new_model.generate_Poisson_2D_finitearea(T=T,M=M)

    func = lambda x : basicfunctions.exponential_kernel(x,alpha=3,beta=3.5)
    NeuronLinear = NeuronLinear(intial_intensity=1,kernel_function= func,poisson_measure=poisson_measure,T=T,M=M)

    plt.figure(0)
    NeuronLinear.intensity_plot("Intensity of A")
    plt.show()

    

