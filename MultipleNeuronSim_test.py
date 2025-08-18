import MultipleNeuronSim as Neurons
import new_model
import basicfunctions

def test_intensity_linear():
    #todo must be more trustworthy and have less out-of-the-blue constants
    poisson_measure = {"time" : [2.5,3.8],
                       "theta" : [12., 23.],
                       "T": 4,
                       "M":24}
    neuron = Neurons.NeuronLinear(initial_intensity=13,
        kernel_function=lambda x : basicfunctions.exponential_kernel(x,alpha=2,beta=4),
        poisson_measure=poisson_measure)
    events = neuron.events()
    assert events[0] == 2.5