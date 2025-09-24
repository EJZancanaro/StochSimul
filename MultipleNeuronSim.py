import new_model
import basicfunctions
import numpy as np
import matplotlib.pyplot as plt

from basicfunctions import ExponentialDecay

import new_model
import new_model_fast
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
    simulation_was_ran = False
    def __init__(self, initial_intensity , list_parent_kernels, poisson_measure):
        self.id = NeuronLinear.number_instances_linear
        NeuronLinear.number_instances_linear+=1
        NeuronLinear.list_linear_neurons.append(self)

        self.initial_intensity = initial_intensity
        self.array_parent_kernels = list_parent_kernels
        self.is_exponential_decay = basicfunctions.is_exponential_decay(self.array_parent_kernels[self.id])
        self.poisson_measure = poisson_measure

        self.T = poisson_measure["T"]
        self.M = poisson_measure["M"]

        self.history = []

    @staticmethod
    def events():

        assert not NeuronLinear.simulation_was_ran

        all_poisson_measuresA = order_poisson_measure(
            [neuron.poisson_measure for neuron in NeuronLinear.list_linear_neurons]
        )
        array_times = all_poisson_measuresA["time"]
        array_thetas = all_poisson_measuresA["theta"]
        array_ids = all_poisson_measuresA["id"]


        array_current_intensities = np.zeros(NeuronLinear.number_instances_linear)
        #array_current_events = [ [] for _ in range(NeuronSemilinear.number_instances_semilinear) ]
        #array_current_events[i] will be the events of neuron i up to our point in simulation


        for neuron in NeuronLinear.list_linear_neurons: #initliasing intensities
            array_current_intensities[neuron.id] = neuron.initial_intensity


        #assert max(array_ids)<=len(NeuronSemilinear.list_semilinear_neurons)
        #otherwise something wrong happened

        #assert array_ids!=np.zeros_like(array_ids) debugging
        for t, theta, neuron_index in zip(array_times,array_thetas,array_ids):

            current_neuron = NeuronLinear.list_linear_neurons[neuron_index]

            array_current_intensities[neuron_index] = sum (
                [ new_model.hawkes_intensity(t,
                    #mu=current_neuron.initial_intensity,
                    mu=0,
                    phi=current_neuron.array_parent_kernels[id_parent_neuron],
                    history = np.array(NeuronLinear.list_linear_neurons[id_parent_neuron].history))
                 for id_parent_neuron in range(NeuronLinear.number_instances_linear)
                ]
            )
            array_current_intensities[neuron_index] *=1/NeuronLinear.number_instances_linear
            array_current_intensities[neuron_index] += current_neuron.initial_intensity

            if theta<max(0, array_current_intensities[neuron_index]) :
                current_neuron.history.append(t)

        NeuronLinear.history_to_array()
        NeuronLinear.simulation_was_ran=True

    @staticmethod
    def history_to_array():
        """
        Turns the history of each neuron, who are lists during their construction, into arrays
        """
        for neuron in NeuronLinear.list_linear_neurons:
            neuron.history = np.array(neuron.history)

    def intensity_values(self):
        """
        Computes the intensity over time of the neuron.
        :return: Array representing the values of the intensity as time progresses up to self.T
        """

        T = self.T
        time_scale = np.linspace(0, T, T * NUMBER_OF_POINTS_PER_UNIT_OF_TIME)
        intensity_array = np.zeros_like(time_scale)

        for (i, t) in enumerate(time_scale):
            intensity_array[i] = sum(
                [new_model.hawkes_intensity(t,
                                            mu=0,
                                            phi=self.array_parent_kernels[id_parent_neuron],
                                            history=np.array(
                                                NeuronLinear.list_linear_neurons[id_parent_neuron].history[
                                                    NeuronLinear.list_linear_neurons[
                                                        id_parent_neuron].history < t
                                                    ]))
                 for id_parent_neuron in range(NeuronLinear.number_instances_linear)
                 ]
            )
            intensity_array[i] *= 1 / NeuronLinear.number_instances_linear
            intensity_array[i] += self.initial_intensity

        return intensity_array, time_scale

    def intensity_plot(self, string):
        """
        Plots the intensity of a given neuron
        :param string: String that will appear in the legend of the plot
        """
        intensity_values, time_scale = self.intensity_values()
        plt.plot(time_scale, intensity_values, label=string)

    @staticmethod
    def array_mean_of_intensities():
        """
        Computes the array containing for all times the mean of the intensities of all linear neurons

        :return: array of the means of intensities of linear neurons over time
        """

        sum_intensities = 0

        assert len(NeuronLinear.list_linear_neurons) != 0
        for linear_neuron in NeuronLinear.list_linear_neurons:
            intensities, time_scale = linear_neuron.intensity_values()
            # remark. It is imperative all time_scale variables throughout the loop are identical.
            sum_intensities += intensities

        return 1 / NeuronLinear.number_instances_linear * sum_intensities , time_scale
    @staticmethod
    def plot_means():
        """
        Plots the mean intensity of all linear neurons
        """
        means, time_scale = NeuronLinear.array_mean_of_intensities()
        plt.plot( time_scale, means, label="Intensity of the mean of all linear neurons")

    @staticmethod
    def reinitialise():
        NeuronLinear.number_instances_linear = 0
        NeuronLinear.list_linear_neurons = []

        #NeuronLinear.simulation_was_ran = False

class NeuronSemilinear() :

    number_instances_semilinear = 0
    list_semilinear_neurons = []

    simulation_was_ran = False

    def __init__(self,initial_intensity,list_parent_kernels,poisson_measure):
        # It is imperative all linear Neurons are defined before any semilinear neurons

        assert NeuronLinear.list_linear_neurons != []

        self.id = NeuronSemilinear.number_instances_semilinear #integer that identifies the neuron
        NeuronSemilinear.number_instances_semilinear+=1
        NeuronSemilinear.list_semilinear_neurons.append(self)

        self.initial_intensity = initial_intensity
        self.T = poisson_measure["T"]
        self.M = poisson_measure["M"]

        self.poisson_measure = poisson_measure

        self.history = []

        self.array_parent_kernels = list_parent_kernels
        self.are_exponential_decays = [basicfunctions.is_exponential_decay(function) for function in self.array_parent_kernels]

        #list indexed by j of the kernels phi^{j->i}, where i is self.id
        #We must not generate the events before creating all semilinear neurons

    @staticmethod
    def events(): # this might actually be way more efficient as a static method because when computing ones events
                      # we already need to compute others, so might as well store them. Though in that case we must becareful that
                      # it is only run once
        """
        Generates and stores all events of all semi-linear neurons progressively. It can and should only be run once.
        """

        assert not NeuronSemilinear.simulation_was_ran

        mean_of_linear_intensities, time_scale = NeuronLinear.array_mean_of_intensities()



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
            if theta<max(0, array_current_intensities[neuron_index] - current_mean_linears ) :
                current_neuron.history.append(t)
        NeuronSemilinear.history_to_array()
        NeuronSemilinear.simulation_was_ran = True


    @staticmethod
    def history_to_array():
        """
        Turns the history of each neuron, who are lists during their construction, into arrays
        """
        for neuron in NeuronSemilinear.list_semilinear_neurons :
            neuron.history = np.array(neuron.history)

    def intensity_values(self):
        """
        Computes the intensity over time of the neuron.
        :return: Array representing the values of the intensity as time progresses up to self.T
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



        return intensity_array , time_scale

    def intensity_plot(self,string):
        """
        Plots the intensity of a given neuron over time. Should be called after a figure has already been created.
        :param string: String that will legend the ploting after calling plt.show()

        """
        intensity_values, time_scale = self.intensity_values()
        plt.plot(time_scale,intensity_values,label=string)

    def plot_poisson_events(self,string):
        """
        Plot on an already open figure the points of the poisson process that generate the interaction of the neuron with others.
        This is done according to whether they ended up being selected or not.
        """
        assert NeuronSemilinear.simulation_was_ran
        #otherwise, we can't chose which ones to plot

        intensity, time_scale =  self.intensity_values()
        #Index where maximum value of intensity is obtained, so we can ignore all thetas bigger than it
        theta_max = max(intensity)

        times = self.poisson_measure["time"]
        thetas = self.poisson_measure["theta"]

        mask_for_low_points = (thetas <= theta_max)

        intensity_at_poisson_times = np.interp(times,time_scale,intensity)

        mean_linear,time_scaleA = NeuronLinear.array_mean_of_intensities()
        assert np.array_equal(time_scaleA, time_scale)
        value_mean_at_poisson = np.interp(times,time_scaleA,mean_linear)

        mask_for_accepted_points = (thetas <= np.maximum(0,intensity_at_poisson_times - value_mean_at_poisson) )
        mask_for_rejected_points = (thetas > np.maximum(0,intensity_at_poisson_times - value_mean_at_poisson) )

        plt.plot(times[mask_for_low_points & mask_for_accepted_points],
                 thetas[mask_for_low_points & mask_for_accepted_points],
                 '+',
                 color='green',
                 label=string + ' (accepted) ')
        plt.plot(times[mask_for_low_points & mask_for_rejected_points],
                 thetas[mask_for_low_points & mask_for_rejected_points],
                 '+',
                 color='red',
                 label=string + ' (rejected) ')
        plt.plot(time_scale , np.maximum(0,intensity-mean_linear) ,
                  label=" Acceptation zone ")

    @staticmethod
    def reinitialise():
        NeuronSemilinear.number_instances_semilinear = 0
        NeuronSemilinear.list_semilinear_neurons = []

        NeuronSemilinear.simulation_was_ran = False



if __name__ == "__main__":
    T = 5
    M = 1000
    poisson_measureA1 = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureA2 = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureA3 = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureB1 = new_model.generate_Poisson_2D_finitearea(T=T,M=M)
    poisson_measureB2 = new_model.generate_Poisson_2D_finitearea(T=T,M=M)

    muA1, muA2, muB1,muB2 = 4,5,10,12

    phiA11 = lambda x: basicfunctions.exponential_kernel(x,alpha=5,beta=6)
    phiA12 = lambda x: basicfunctions.exponential_kernel(x,alpha=3, beta=10)
    phiA21 = lambda x: basicfunctions.exponential_kernel(x,alpha=2, beta=3)
    phiA22 = lambda x: basicfunctions.exponential_kernel(x,alpha=1, beta=9)

    phiB11 = lambda x: basicfunctions.exponential_kernel(x, alpha=7, beta=7)
    phiB21 = lambda x: basicfunctions.exponential_kernel(x, alpha=2, beta=3)
    phiB12 = lambda x: basicfunctions.exponential_kernel(x, alpha=3, beta=4)
    phiB22 = lambda x: basicfunctions.exponential_kernel(x, alpha=8, beta=6)

    A1 = NeuronLinear(initial_intensity=muA1, list_parent_kernels=[phiA11,phiA21], poisson_measure=poisson_measureA1)
    A2 = NeuronLinear(initial_intensity=muA2, list_parent_kernels=[phiA12,phiA22], poisson_measure=poisson_measureA2)

    parent_kernels_B1 = [phiB11,phiB21]
    parent_kernels_B2 = [phiB12,phiB22]

    B1 = NeuronSemilinear(initial_intensity=muB1, list_parent_kernels=[phiB11,phiB21], poisson_measure=poisson_measureB1)
    B2 = NeuronSemilinear(initial_intensity=muB2, list_parent_kernels=[phiB12,phiB22], poisson_measure=poisson_measureB2)

    NeuronLinear.events()
    NeuronSemilinear.events()

    plt.figure()

    A1.intensity_plot(string="Intensity of A1")
    A2.intensity_plot(string="Intensity of A2")

    #B1.intensity_plot(string="Intensity of B1")
    #B2.intensity_plot(string="Intensity of B2")


    NeuronLinear.plot_means()

    #B1.plot_poisson_events("Poisson events of B1")
    plt.legend(fontsize="xx-small")
    plt.tight_layout()
    plt.show()



