import new_model
import basicfunctions
import numpy as np
import matplotlib.pyplot as plt

from projet_stage.StochSimul.new_model import generate_Poisson_2D_finitearea, hawkes_intensity

NUMBER_OF_POINTS_PER_UNIT_OF_TIME = 10**3

#TODO Becareful, during development we switched between dictionary representation of a poisson measure
# and a class representation. This creates inconsistencies that are prone to bugs.
#Consensus : ALL will be dictionnaries



def order_poisson_measure(list_poisson_measure_dicts):

    """
    Takes as input a list of poisson_measures and returns a poisson_measured dictionary in such a way that the times are
    ordered and the thetas correspond to the times
    :param list_poisson_measure_dict:
    :return:
    """

    #I can imagine a usecase where this is not necessary, but just in case
    T = list_poisson_measure_dicts[0]["T"]
    M = list_poisson_measure_dicts[0]["M"]
    for dict in list_poisson_measure_dicts:
        assert T == dict["T"]

    all_times = []
    all_thetas = []
    all_origins = []

    for index, poisson_dict in enumerate(list_poisson_measure_dicts):
        number_2D_points = len(poisson_dict["time"])
        all_times.append(poisson_dict["time"])
        all_thetas.append(poisson_dict["theta"])
        all_origins.append(np.full(shape=number_2D_points, fill_value=index, dtype=int))

    # Concatenate into single arrays
    all_times = np.concatenate(all_times)
    all_thetas = np.concatenate(all_thetas)
    all_origins = np.concatenate(all_origins)

    # Sort by time
    sorted_indexes = np.argsort(all_times)

    return {
        "time": all_times[sorted_indexes],
        "theta": all_thetas[sorted_indexes],
        "T": T,
        "M": M,
        "id": all_origins[sorted_indexes]
    }

# class Poisson_measure():
#     list_Poisson_measures = []
#     final = False
#     def __init__(self,T,M,poisson_dict=None):
#
#         assert not Poisson_measure.final
#
#         self.T = T
#         self.M = M
#
#         if poisson_dict is None : #If a poisson dictionary is not given, we generate one
#                                   #otherwise use the given one
#             poisson_dict = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
#
#         self.time = poisson_dict["time"]
#         self.theta = poisson_dict["theta"]
#
#         self.id = len(Poisson_measure.list_Poisson_measures)
#         Poisson_measure.list_Poisson_measures.append(poisson_dict)
#     @staticmethod
#     def order_all():
#
#         """
#         Return a Poisson_measure instance that contains all 2D points of all Poisson_measures ordered by time
#         Side-effect : blocks all future Poisson_measure instance creations
#         :return: Poisson_measure object
#         """
#
#         ordered_poisson_dict = order_poisson_measure(Poisson_measure.list_Poisson_measures)
#         #return_object = Poisson_measure(T = None, M=None,poisson_dict=ordered_poisson_dict)
#         Poisson_measure.final = True
#
#         return ordered_poisson_dict
#
# class Multiple_poisson_measures():

    # def __init__(self,list_poisson_measure_dicts):
    #
    #     ordered_results = order_poisson_measure(list_poisson_measure_dicts)
    #     self.array_ids =  ordered_results["id"]
    #     self.array_times = ordered_results["time"]
    #     self.array_thetas = ordered_results["theta"]



class NeuronLinear():
    number_instances_linear = 0
    list_linear_neurons = []
    def __init__(self, initial_intensity , kernel_function, poisson_measure,T,M):
        self.id = NeuronLinear.number_instances_linear
        NeuronLinear.number_instances_linear+=1
        NeuronLinear.list_linear_neurons.append(self)

        self.initial_intensity = initial_intensity
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

    simulation_was_ran = False

    def __init__(self,initial_intensity,list_parent_kernels,T,M,poisson_measure):
        assert NeuronLinear.list_linear_neurons != []
        #It is imperative all linear Neurons are defined before any semilinear neurons

        self.id = NeuronSemilinear.number_instances_semilinear #integer that identifies the neuron
        NeuronSemilinear.number_instances_semilinear+=1
        NeuronSemilinear.list_semilinear_neurons.append(self)

        self.initial_intensity = initial_intensity
        self.T = T
        self.M = M

        self.poisson_measure = poisson_measure

        self.history = []
        self.array_parent_kernels = list_parent_kernels #list indexed by j of the kernels phi^{j->i}, where i is self.id
        #We must not generate the events before creating all semilinear neurons

    def inform(self,array_influence_functions,array_inhibiting, array_exciting, array_poisson_measure_exciting):
        #Fills in the previously unfilled attributes of the object

        self.array_influence_functions = np.copy(array_influence_functions)
        self.array_exciting = np.copy(array_exciting)
        self.array_inhibiting = np.copy(array_inhibiting)
        self.array_poisson_measure_exciting = np.copy(array_poisson_measure_exciting)


    @staticmethod
    def events(): # this might actually be way more efficient as a static method because when computing ones events
                      # we already need to compute others, so might as well store them. Though in that case we must becareful that
                      # it is only run once
        """
        Generates and stores all events of all semi-linear neurons progressively. It can and should only be run once.
        """

        assert not NeuronSemilinear.simulation_was_ran

        sum_intensities = 0
        ##############Computing the mean of the LINEAR intensities
        assert len(NeuronLinear.list_linear_neurons)!=0
        for linear_neuron in NeuronLinear.list_linear_neurons:

            intensities, time_scale = linear_neuron.intensity_values()
            #remark. It is imperative all time_scale variables throughout the loop are identical.
            sum_intensities += intensities
        print("Sum_intensities ", sum_intensities)
        mean_of_linear_intensities = 1/NeuronLinear.number_instances_linear * sum_intensities
        #####################
        ############# Getting for every semilinear neuron their events

        #all_poisson_measuresB = Poisson_measure.order_all()
                      #TODO HERE IS THE ERROR, this takes ALL PM, we only want the ones generating Semilinear

        all_poisson_measuresB = order_poisson_measure(
            [neuron.poisson_measure for neuron in NeuronSemilinear.list_semilinear_neurons]
        )
        array_times = all_poisson_measuresB["time"]
        array_thetas = all_poisson_measuresB["theta"]
        array_ids = all_poisson_measuresB["id"]


        array_current_intensities = np.zeros(NeuronSemilinear.number_instances_semilinear)
        #array_current_events = [ [] for _ in range(NeuronSemilinear.number_instances_semilinear) ]
        #array_current_events[i] will be the events of neuron i up to our point in simulation


        for neuron in NeuronSemilinear.list_semilinear_neurons: #initliasing intensities
            array_current_intensities[neuron.id] = neuron.initial_intensity


        #assert max(array_ids)<=len(NeuronSemilinear.list_semilinear_neurons)
        #otherwise something wrong happened

        #assert array_ids!=np.zeros_like(array_ids) debugging
        for t, theta, neuron_index in zip(array_times,array_thetas,array_ids):

            current_neuron = NeuronSemilinear.list_semilinear_neurons[neuron_index]

            array_current_intensities[neuron_index] = sum (
                [ new_model.hawkes_intensity(t,
                    #mu=current_neuron.initial_intensity,
                    mu=0,
                    phi=current_neuron.array_parent_kernels[id_parent_neuron],
                    history = np.array(NeuronSemilinear.list_semilinear_neurons[id_parent_neuron].history))
                 for id_parent_neuron in range(NeuronSemilinear.number_instances_semilinear)
                ]
            )
            array_current_intensities[neuron_index] *=1/NeuronSemilinear.number_instances_semilinear
            array_current_intensities[neuron_index] += current_neuron.initial_intensity

            current_mean_linears = np.interp(t, time_scale, mean_of_linear_intensities)
            print("Theta: ",theta)
            print("upper bound ",max(0, array_current_intensities[neuron_index] - current_mean_linears))
            if theta<max(0, array_current_intensities[neuron_index] - current_mean_linears ) :
                current_neuron.history.append(t)
        NeuronSemilinear.history_to_array()
        NeuronSemilinear.simulation_was_ran = True

    @staticmethod
    def history_to_array():
        for neuron in NeuronSemilinear.list_semilinear_neurons :
            neuron.history = np.array(neuron.history)

    def intensity_values(self):
        #Same as the simple case, we suppose we know all events up to time t-, what is the intensity at time t?
        """
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
        """
        T = self.T
        time_scale = np.linspace(0,T,T*NUMBER_OF_POINTS_PER_UNIT_OF_TIME)
        intensity_array = np.zeros_like(time_scale)


        for (i,t) in enumerate(time_scale):
            intensity_array[i] = sum (
                [ new_model.hawkes_intensity(t,
                    mu=0,
                    phi=self.array_parent_kernels[id_parent_neuron],
                    history = np.array(NeuronSemilinear.list_semilinear_neurons[id_parent_neuron].history[
                                        NeuronSemilinear.list_semilinear_neurons[id_parent_neuron].history<t
                                       ]))
                 for id_parent_neuron in range(NeuronSemilinear.number_instances_semilinear)
                ]
            )
            intensity_array[i] *= 1 / NeuronSemilinear.number_instances_semilinear
            intensity_array[i] += self.initial_intensity



        #assert False #TODO this implementation is false. The list of all kernel functions has to be a matrix.
                     # The current state of the implementation considers that neurons interact with others
                     # with a kernel that is Identical to the one he uses to interact with itself.
                     # This is not true in the general case.

                     #TODO How to solve it : instead of kernel_function as an attribute, every instance
                     # must have as an attribute a  list of kernels, that MUST be the same size as
                     # the list of all existing semilinear neurons.
                     # to acces his own kernel, a neuron need only to acces that list at the index of his own id.

        #TODO Take inspiration from the function events()
        return intensity_array , time_scale

    def intensity_plot(self,string):
        intensity_values, time_scale = self.intensity_values()
        plt.plot(time_scale,intensity_values,label=string)


if __name__ == "__main__":
    T = 5
    M = 1000
    poisson_measureA1 = generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureA2 = generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureA3 = generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureB1 = generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureB2 = generate_Poisson_2D_finitearea(T=T,M=M)

    muA1, muA2, muA3, muB1,muB2 = 1,2,3,13,25

    phiA1 = lambda x: basicfunctions.exponential_kernel(x,alpha=4,beta=6)
    phiA2 = lambda x: basicfunctions.exponential_kernel(x,alpha=3.5, beta=4)
    phiA3 = lambda x: basicfunctions.exponential_kernel(x,alpha=3.9, beta=4)

    phiB2 = lambda x: basicfunctions.exponential_kernel(x, alpha=8.5, beta=9)
    phiB11 = lambda x: 0
    phiB21 = lambda x: basicfunctions.exponential_kernel(x, alpha=2, beta=3)
    phiB12 = lambda x: basicfunctions.exponential_kernel(x, alpha=2, beta=3)
    phiB22 = lambda x: 0

    A1 = NeuronLinear(initial_intensity=muA1, kernel_function=phiA1, poisson_measure=poisson_measureA1,T=T,M=M)
    A2 = NeuronLinear(initial_intensity=muA2, kernel_function=phiA2, poisson_measure=poisson_measureA2, T=T, M=M)
    A3 = NeuronLinear(initial_intensity=muA3, kernel_function=phiA3, poisson_measure=poisson_measureA3, T=T, M=M)

    parent_kernels_B1 = [phiB11,phiB21]
    parent_kernels_B2 = [phiB12,phiB22]

    B1 = NeuronSemilinear(initial_intensity=muB1, list_parent_kernels=[phiB11,phiB21], poisson_measure=poisson_measureB1,T=T,M=M)
    B2 = NeuronSemilinear(initial_intensity=muB2, list_parent_kernels=[phiB12,phiB22], poisson_measure=poisson_measureB2,T=T,M=M)

    NeuronSemilinear.events()

    plt.figure()
    print(A1.history)
    print(A2.history)
    print(A3.history)
    print(B1.history)
    print(B2.history)
    A1.intensity_plot(string="Intensity of A1")
    A2.intensity_plot(string="Intensity of A2")
    A3.intensity_plot(string="Intensity of A3")

    B1.intensity_plot(string="Intensity of B1")
    B2.intensity_plot(string="Intensity of B2")
    plt.legend()
    plt.show()



