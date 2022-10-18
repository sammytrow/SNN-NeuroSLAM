import math
import pyNN.brian2 as p
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel
from Simulation.log_data.training_data.SNN_training_data5_laps import velocity, cur_pos, landmarks
from Simulation.data_management import *
from datetime import datetime
filename = 'snn_test'

def calc_vision_cone(endv, pos, angle):
    endv = list(endv)
    endv[0] -= pos[0]
    endv[1] -= pos[1]

    calcx = endv[0] * math.cos(angle) - endv[1] * math.sin(angle)
    calcy = endv[0] * math.sin(angle) + endv[1] * math.cos(angle)

    endv[0] = calcx + pos[0]
    endv[1] = calcy + pos[1]
    return endv

start_dir = (1,0)
class mapping_SNN:
    def __init__(self):
        self.parameters = {'cm': 0.25,
                           'i_offset': 0.0,
                           'tau_m': 10.0,
                           'tau_refrac': 2.0,
                           'tau_syn_E': 5.0,
                           'tau_syn_I': 5.0,
                           'v_reset': -70.0,
                           'v_rest': -70.0,
                           'v_thresh': -50.0
                           }
        self.input_cp = cur_pos
        self.pos_neurons = 3
        self.landmark_data = landmarks
        self.number_of_landmarks = 300#163

        self.input_FOV_x = []
        self.input_FOV_y = []
        self.FOV_x_neurons = 141  # plus 40 for noise or incorrect calculation
        self.FOV_y_neurons = 211  # plus 10 for noise
        # self.number_of_landmarks = 400
        self.seen_landmark = []
        self.y_dist = []
        for tim in landmarks:
            x_input = []
            y_input = []
            y_distance = []
            for item in tim[1]:
                if item[1] == 0 or item[1] == 1:
                    x_input.append(abs(item[-2]))
                    y_v = item[-1]
                    y_distance.append(y_v)
                    if y_v == 0:
                        Y_n = self.FOV_y_neurons / 2
                    elif y_v < 0:
                        Y_n = (self.FOV_y_neurons / 2) + abs(y_v)
                    else:
                        Y_n = (self.FOV_y_neurons / 2) - y_v
                    y_input.append(Y_n)
                    if item[-2] > 110 or item[-2] < 0:
                        print("x: ", item[-2])
            self.input_FOV_x.append(x_input)
            self.input_FOV_y.append(y_input)

        self.data_elements = self.input_cp[-1][0]
        print('data_elements: ', self.data_elements)
        self.data_elements = 2000

        self.delay = 0.5
        self.epoch = 3
        self.learning = 2.0
        self.firing_period = 2
        self.epoch_time = self.firing_period * self.data_elements
        self.traintime = self.epoch_time * self.epoch

        self.populations_m = list()
        # quadrature phase shift keying

        self.output_theta_neurouns = 73
        self.output_num_realistic_locations_x = 251  # 1200/5
        self.output_num_realistic_locations_y = 131  # 600/5

    def gen_pop(self):
        self.populations_m.append(
            p.Population(self.output_num_realistic_locations_x, p.IF_cond_exp(**self.parameters),label='input x pos'))
        self.populations_m.append(
            p.Population(self.output_num_realistic_locations_y, p.IF_cond_exp(**self.parameters), label='input y pos'))
        self.populations_m.append(
            p.Population(self.number_of_landmarks, p.IF_cond_exp(**self.parameters), label='number landmarks'))

        self.populations_m.append(
            p.Population(self.FOV_x_neurons, p.IF_cond_exp(**self.parameters), label='input FOV x'))
        self.populations_m.append(
            p.Population(self.FOV_y_neurons, p.IF_cond_exp(**self.parameters), label='input FOV y'))
        self.populations_m.append(
            p.Population(self.output_num_realistic_locations_x, p.IF_cond_exp(**self.parameters),
                         label='landmark x pos'))
        self.populations_m.append(
            p.Population(self.output_num_realistic_locations_y, p.IF_cond_exp(**self.parameters),
                         label='landmark y pos'))


    def train_snn(self):
        start_date_time = datetime.now()
        print("timestamp: ", start_date_time)
        p.setup(timestep=1, min_delay=1, max_delay=1)
        self.gen_pop()

        self.train_gen_spike_input()

        tau_plus = 15.0
        tau_minus = 25.0
        A_plus = 0.01
        A_minus = 0.012
        w_min = 0.0
        w_max = 0.05
        stdp = p.STDPMechanism(
            timing_dependence=p.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus,
                                              A_plus=A_plus, A_minus=A_minus),
            weight_dependence=p.AdditiveWeightDependence(w_min=w_min, w_max=w_max),
            dendritic_delay_fraction=0,
            weight=RandomDistribution('normal', mu=0.000005, sigma=0.000001),
            delay=1.0)

        synapse = p.Projection(self.populations_m[0], self.populations_m[2], p.AllToAllConnector(), synapse_type=stdp)
        synapse2 = p.Projection(self.populations_m[1], self.populations_m[2], p.AllToAllConnector(), synapse_type=stdp)

        p.Projection(self.populations_m[0], self.populations_m[3], p.AllToAllConnector())
        p.Projection(self.populations_m[1], self.populations_m[4], p.AllToAllConnector())
        synapse3 = p.Projection(self.populations_m[3], self.populations_m[5], p.AllToAllConnector(), synapse_type=stdp)
        synapse4 = p.Projection(self.populations_m[4], self.populations_m[6], p.AllToAllConnector(), synapse_type=stdp)

        self.record_data()

        print('start - train_time: ', self.traintime)

        p.run(self.traintime)

        runtime = datetime.now() - start_date_time
        runtime_sec = runtime.total_seconds()
        print('runtime seconds: ', runtime_sec)
        minutes = divmod(runtime_sec, 60)[0]
        print('runtime minutes: ', minutes)

        date = save_date()
        synapse.save('weight', 'weights/mapping/'+date+'train_mapping_landmark_weights.txt', format='list')
        synapse2.save('weight', 'weights/mapping/'+date+'train_mapping_landmark_weights2.txt', format='list')
        synapse3.save('weight', 'weights/mapping/'+date+'train_mapping_landmark_weights_loc_x.txt', format='list')
        synapse4.save('weight', 'weights/mapping/'+date+'train_mapping_landmark_weights_loc_y.txt', format='list')

        self.synapseWeights = synapse.get(["weight"], format="list")
        self.synapseWeights2 = synapse2.get(["weight"], format="list")
        self.synapseWeights3 = synapse3.get(["weight"], format="list")
        self.synapseWeights4 = synapse4.get(["weight"], format="list")


        spike_count = self.populations_m[0].get_spike_counts()
        print("spike_count x pos: ", spike_count)
        mean = self.populations_m[0].mean_spike_count()
        print("mean x: ", mean)
        spike_count = self.populations_m[1].get_spike_counts()
        print("spike_count y pos: ", spike_count)
        mean = self.populations_m[1].mean_spike_count()
        print("mean y: ", mean)

        spike_count = self.populations_m[2].get_spike_counts()
        print("spike_count: ", spike_count)
        mean = self.populations_m[2].mean_spike_count()
        print("mean : ", mean)

        self.populations_m = list()

    def train_gen_spike_input(self):
        nur_num = 0
        id_list = []
        index_list = []
        string_list = []
        for i in range(self.data_elements - 1):
            for j in range(0, self.epoch):
                spiketime = self.firing_period + (i * self.firing_period) + (j * self.epoch_time)
                endspiketime = self.firing_period + ((i + 1) * self.firing_period) + (j * self.epoch_time)

                x_n = round(self.input_cp[i][1][0] / 5)
                y_n = round(self.input_cp[i][1][1] / 5)

                pulse = p.DCSource(amplitude=10, start=spiketime, stop=endspiketime)
                self.populations_m[0][round(x_n)].inject(pulse)

                pulse = p.DCSource(amplitude=10, start=spiketime, stop=endspiketime)
                self.populations_m[1][round(y_n)].inject(pulse)
                string = 'loc- X:'+str(x_n)+' Y:'+str(y_n)+' - seen landmarks: '

                for x_value in self.input_FOV_x[i]:
                    pulse = p.DCSource(amplitude=20, start=spiketime, stop=endspiketime)
                    self.populations_m[3][round(x_value)].inject(pulse)
                    x_p = x_n + round(x_value/5)
                    if x_p >=  self.output_num_realistic_locations_x:
                        x_p = self.output_num_realistic_locations_x-1
                    pulse = p.DCSource(amplitude=20, start=endspiketime, stop=(endspiketime + self.learning))
                    self.populations_m[5][x_p].inject(pulse)

                for y_value in self.input_FOV_y[i]:
                    pulse = p.DCSource(amplitude=20, start=spiketime, stop=endspiketime)
                    self.populations_m[4][round(y_value)].inject(pulse)
                    if (self.FOV_y_neurons / 2) < y_value:
                        y_fov = round((y_value - (self.FOV_y_neurons / 2))/5)
                    elif (self.FOV_y_neurons / 2) == y_value:
                        y_fov = 0
                    else:
                        y_fov = -round(y_value/ 5)
                    y_p = y_n + y_fov
                    pulse = p.DCSource(amplitude=20, start=endspiketime, stop=(endspiketime + self.learning))
                    self.populations_m[6][y_p].inject(pulse)

                for l in self.landmark_data[i][1]:
                    if l[0] in id_list:
                        index = id_list.index(l[0])
                        pulse = p.DCSource(amplitude=20, start=endspiketime, stop=(endspiketime + self.learning))
                        self.populations_m[2][index].inject(pulse)
                        index_list.append(index)
                        string += ''+str(index)+', '
                    else:
                        pulse = p.DCSource(amplitude=20, start=endspiketime, stop=(endspiketime + self.learning))
                        self.populations_m[2][nur_num].inject(pulse)
                        id_list.append(l[0])
                        nur_num += 1
                        index_list.append(nur_num)
                        string += '' + str(nur_num) + ', '
                string += '-----end '
                string_list.append(string)
        print("finished")
        #print('self.seen_landmark: ', index_list)
        print("num seen_landmark: ", len(index_list))
        #print('string_list: ', string_list)

    def plot_all_results(self, title):
        plot_landmarks(self.populations_m[2], title)
        #plot_landmarks_pos(self.populations_m[5], title)
        #plot_landmarks_pos(self.populations_m[6], title)

        plot_result(self.populations_m[0], "" + title + "_MAPPING_input_pos_and_velocity" + str(self.data_elements - 1) + "", "mapping/")
        plot_result(self.populations_m[1], "" + title + "_MAPPING_input_landmark_x_fov" + str(self.data_elements - 1) + "", "mapping/")
        plot_result(self.populations_m[2], "" + title + "_MAPPING_input_landmark_y_fov" + str(self.data_elements - 1) + "", "mapping/")

        plot_results3(self.populations_m[0], self.populations_m[1], title, "mapping/")

        save_results(self.populations_m[0], "results/mapping/" + title + "input_pos")
        save_results(self.populations_m[1], "results/mapping/" + title + "input_landmark_fov_x")
        save_results(self.populations_m[2], "results/mapping/" + title + "input_landmark_fov_y")

    def record_data(self):
        self.populations_m[0][0:4].record(['v', 'gsyn_exc'])
        self.populations_m[0].record('spikes')
        self.populations_m[1][0:10].record(['v', 'gsyn_exc'])
        self.populations_m[1].record('spikes')
        self.populations_m[2][0:10].record(['v', 'gsyn_exc'])
        self.populations_m[2].record('spikes')
        self.populations_m[3][0:10].record(['v', 'gsyn_exc'])
        self.populations_m[3].record('spikes')
        self.populations_m[4][0:10].record(['v', 'gsyn_exc'])
        self.populations_m[4].record('spikes')
        self.populations_m[5][0:10].record(['v', 'gsyn_exc'])
        self.populations_m[5].record('spikes')
        self.populations_m[6][0:10].record(['v', 'gsyn_exc'])
        self.populations_m[6].record('spikes')

    def output_connections(self):
        outputConnectors = []
        x_inputs = self.output_num_realistic_locations_x - 1
        y_inputs = self.output_num_realistic_locations_y - 1

        cardinal_weight_1 = 0.009
        horizontal_weight_1 = 0.05
        cardinal_weight_2 = 0.006
        car_hori_weight_2 = 0.001
        horizontal_weight_2 = 0.002

        for i in range(x_inputs):
            for j in range(y_inputs):
                outputConnectors.append((j, i, 0.1, self.delay))
                outputConnectors.append((i, j, 0.1, self.delay))
                outputConnectors.append((i, j + 1, cardinal_weight_1, self.delay))
                outputConnectors.append((i + 1, j, cardinal_weight_1, self.delay))
                outputConnectors.append((i + 1, j + 1, horizontal_weight_1, self.delay))

                if i > 0:
                    if j <= (y_inputs - 2):
                        outputConnectors.append((i - 1, j + 2, car_hori_weight_2, self.delay))
                    if j <= (y_inputs - 1):
                        outputConnectors.append((i - 1, j + 1, horizontal_weight_1, self.delay))
                    outputConnectors.append((i - 1, j, cardinal_weight_1, self.delay))
                if j > 0:
                    if i <= (x_inputs - 2):
                        outputConnectors.append((i + 2, j - 1, car_hori_weight_2, self.delay))
                    if i <= (x_inputs - 1):
                        outputConnectors.append((i + 1, j - 1, horizontal_weight_1, self.delay))
                    outputConnectors.append((i, j - 1, cardinal_weight_1, self.delay))

                if i > 0 and j > 0:
                    outputConnectors.append((i - 1, j - 1, horizontal_weight_1, self.delay))

                if j > 1:
                    outputConnectors.append((i, j - 2, cardinal_weight_2, self.delay))
                if i <= (x_inputs - 2) and j > 1:
                    outputConnectors.append((i + 2, j - 2, horizontal_weight_2, self.delay))
                if i <= (x_inputs - 1) and j > 1:
                    outputConnectors.append((i + 1, j - 2, car_hori_weight_2, self.delay))
                if i > 0 and j > 1:
                    outputConnectors.append((i - 1, j - 2, car_hori_weight_2, self.delay))

                if i > 1 and j > 1:
                    outputConnectors.append((i - 2, j - 2, horizontal_weight_2, self.delay))

                if i > 1:
                    outputConnectors.append((i - 2, j, cardinal_weight_2, self.delay))
                if j <= (y_inputs - 2) and i > 1:
                    outputConnectors.append((i - 2, j + 2, horizontal_weight_2, self.delay))
                if j <= (y_inputs - 1 and i > 1):
                    outputConnectors.append((i - 2, j + 1, car_hori_weight_2, self.delay))

                if i > 1 and j > 0:
                    outputConnectors.append((i - 2, j - 1, car_hori_weight_2, self.delay))

                if j <= (y_inputs - 2) and i <= (x_inputs - 2):
                    outputConnectors.append((i + 2, j + 2, horizontal_weight_2, self.delay))
                if j <= (y_inputs - 1) and i > 1:
                    outputConnectors.append((i - 2, j + 1, car_hori_weight_2, self.delay))

                if j <= (y_inputs - 2) and i <= (x_inputs - 1):
                    outputConnectors.append((i + 1, j + 2, car_hori_weight_2, self.delay))
                if j <= (y_inputs - 1) and i <= (x_inputs - 2):
                    outputConnectors.append((i + 2, j + 1, car_hori_weight_2, self.delay))
                if j <= (y_inputs - 2):
                    outputConnectors.append((i, j + 2, cardinal_weight_2, self.delay))
                if i <= (x_inputs - 2):
                    outputConnectors.append((i + 2, j, cardinal_weight_2, self.delay))

        return outputConnectors

    def FOV_connections(self):
        outputConnectors = []
        x_FOV = self.FOV_x_neurons - 1 #131
        y_FOV = self.FOV_y_neurons - 1 # 211
        center_y = y_FOV/2

        center_weight_1 = 0.05
        outer_weight_1 = 0.009

        for x in range(x_FOV):
            num_y = 2*x
            if num_y < (x+5):
                num_y = x+5
            elif num_y > y_FOV:
                num_y = y_FOV
            for y in range(int(round((num_y/2)))):
                if ((y/(num_y/2))*100) > 95:
                    outputConnectors.append((x, (center_y-y), outer_weight_1, self.delay))
                else:
                    outputConnectors.append((x, (center_y+y), center_weight_1, self.delay))
        return outputConnectors

    def test_snn(self, load = False):
        p.reset()
        start_date_time = datetime.now()
        print("timestamp: ", start_date_time)
        p.setup(timestep=1, min_delay=1, max_delay=1)

        self.gen_pop()
        self.test_gen_spike()

        if load:
            landmark_connections = p.FromFileConnector("weights/mapping/test_train_mapping_landmark_weights.txt")
            landmark_connections2 = p.FromFileConnector("weights/mapping/test_train_mapping_landmark_weights2.txt")
            landmark_connections3 = p.FromFileConnector("weights/mapping/test_train_mapping_landmark_weights_loc_x.txt")
            landmark_connections4 = p.FromFileConnector("weights/mapping/test_train_mapping_landmark_weights_loc_y.txt")
        else:
            connector = []
            for synapseOffset in range(0, len(self.synapseWeights)):
                fromNeuron = self.synapseWeights[synapseOffset][0]
                toNeuron = self.synapseWeights[synapseOffset][1]
                weight = self.synapseWeights[synapseOffset][2]
                connector = connector + [(fromNeuron, toNeuron, weight, 0.1)]
            landmark_connections = p.FromListConnector(connector)

            connector = []
            for synapseOffset in range(0, len(self.synapseWeights2)):
                fromNeuron = self.synapseWeights2[synapseOffset][0]
                toNeuron = self.synapseWeights2[synapseOffset][1]
                weight = self.synapseWeights2[synapseOffset][2]
                connector = connector + [(fromNeuron, toNeuron, weight, 0.1)]
            landmark_connections2 = p.FromListConnector(connector)
            connector = []
            for synapseOffset in range(0, len(self.synapseWeights3)):
                fromNeuron = self.synapseWeights3[synapseOffset][0]
                toNeuron = self.synapseWeights3[synapseOffset][1]
                weight = self.synapseWeights3[synapseOffset][2]
                connector = connector + [(fromNeuron, toNeuron, weight, 0.1)]
            landmark_connections3 = p.FromListConnector(connector)

            connector = []
            for synapseOffset in range(0, len(self.synapseWeights4)):
                fromNeuron = self.synapseWeights4[synapseOffset][0]
                toNeuron = self.synapseWeights4[synapseOffset][1]
                weight = self.synapseWeights4[synapseOffset][2]
                connector = connector + [(fromNeuron, toNeuron, weight, 0.1)]
            landmark_connections4 = p.FromListConnector(connector)

        p.Projection(self.populations_m[0], self.populations_m[2], landmark_connections)
        p.Projection(self.populations_m[1], self.populations_m[2], landmark_connections2)

        p.Projection(self.populations_m[0], self.populations_m[3], p.AllToAllConnector())
        p.Projection(self.populations_m[1], self.populations_m[4], p.AllToAllConnector())
        p.Projection(self.populations_m[3], self.populations_m[5], landmark_connections3)
        p.Projection(self.populations_m[4], self.populations_m[6], landmark_connections4)

        self.record_data()

        print('start - test_time: ', self.epoch_time)

        p.run(self.epoch_time)

        runtime = datetime.now() - start_date_time
        runtime_sec = runtime.total_seconds()
        print('runtime seconds: ', runtime_sec)
        minutes = divmod(runtime_sec, 60)[0]
        print('runtime minutes: ', minutes)

        self.plot_all_results('test')

    def test_gen_spike(self):
        for i in range(self.data_elements - 1):
            spiketime = self.firing_period + (i * self.firing_period)
            endspiketime = self.firing_period + ((i + 1) * self.firing_period)

            x_n = round(self.input_cp[i][1][0] / 5)
            y_n = round(self.input_cp[i][1][1] / 5)
            pulse = p.DCSource(amplitude=10, start=spiketime, stop=endspiketime)
            self.populations_m[0][round(x_n)].inject(pulse)

            pulse = p.DCSource(amplitude=10, start=spiketime, stop=endspiketime)
            self.populations_m[1][round(y_n)].inject(pulse)

            for x_value in self.input_FOV_x[i]:
                pulse = p.DCSource(amplitude=20, start=spiketime, stop=endspiketime)
                self.populations_m[3][round(x_value)].inject(pulse)

            for y_value in self.input_FOV_y[i]:
                pulse = p.DCSource(amplitude=20, start=spiketime, stop=endspiketime)
                self.populations_m[4][round(y_value)].inject(pulse)

    def initialize_snn(self):
        #p.setup(timestep=1, min_delay=1, max_delay=1)
        self.gen_pop()

        weights1 = p.FromFileConnector("SNN/weights/mapping/train_mapping_landmark_weights.txt")
        weights2 = p.FromFileConnector("SNN/weights/mapping/train_mapping_landmark_weights2.txt")
        weights3 = p.FromFileConnector("SNN/weights/mapping/train_mapping_landmark_weights_loc_x.txt")
        weights4 = p.FromFileConnector("SNN/weights/mapping/train_mapping_landmark_weights_loc_y.txt")

        p.Projection(self.populations_m[0], self.populations_m[2], weights1)
        p.Projection(self.populations_m[1], self.populations_m[2], weights2)
        p.Projection(self.populations_m[0], self.populations_m[3], p.AllToAllConnector())
        p.Projection(self.populations_m[1], self.populations_m[4], p.AllToAllConnector())
        p.Projection(self.populations_m[3], self.populations_m[5], weights3)
        p.Projection(self.populations_m[4], self.populations_m[6], weights4)

        self.record_data()

        self.step_time = 0
        self.step_counter = 0

        self.step_time = 0 #self.start_time

    def run_snn_map(self, data, have_spike = False):
        if not have_spike:
            self.spike_gen(data)

        self.record_data()

        p.run(self.firing_period)

        spike, num = self.get_primary_landmarks(self.populations_m[2])
        print("Neuron: " +str(spike)+ " - number of spikes: "+str(num))


    def spike_gen(self, data):
        self.input_cp = data[3]
        self.input_v = data[2]

        self.landmark_data = data[1]
        self.y_dist = []
        x_input = []
        y_input = []
        y_distance = []
        for item in self.landmark_data:
            if item[1] == 0 or item[1] == 1:
                x_input.append(abs(item[-2]))
                y_v = item[-1]
                y_distance.append(y_v)
                if y_v == 0:
                    Y_n = self.FOV_y_neurons / 2
                elif y_v < 0:
                    Y_n = (self.FOV_y_neurons / 2) + abs(y_v)
                else:
                    Y_n = (self.FOV_y_neurons / 2) - y_v
                y_input.append(Y_n)
                if item[-2] > 110 or item[-2] < 0:
                    print("x: ", item[-2])
        self.input_FOV_x = x_input
        self.input_FOV_y = y_input

        i = 0
        spiketime = self.step_time
        endspiketime = self.step_time + self.firing_period

        self.step_time = endspiketime

        x_n = round(self.input_cp[0] / 5)
        y_n = round(self.input_cp[1] / 5)
        pulse = p.StepCurrentSource(times=[spiketime, endspiketime], amplitudes=[20,0])
        self.populations_m[0][round(x_n)].inject(pulse)

        pulse = p.StepCurrentSource(times=[spiketime, endspiketime], amplitudes=[20,0])
        self.populations_m[1][round(y_n)].inject(pulse)

        for x_value in self.input_FOV_x:
            pulse = p.StepCurrentSource(times=[spiketime, endspiketime], amplitudes=[20,0])
            self.populations_m[3][round(x_value)].inject(pulse)

        for y_value in self.input_FOV_y:
            pulse = p.StepCurrentSource(times=[spiketime, endspiketime], amplitudes=[20,0])
            self.populations_m[4][round(y_value)].inject(pulse)

        self.step_counter += 1

    def get_primary_landmarks(self, pops):
        spikes = pops.get_data('spikes')
        spikes = spikes.segments[0]
        spike_neuron = 0
        spike_number = 0
        for ind, spiketrain in enumerate(spikes.spiketrains):
            counter = 0
            for spike in spiketrain:
                if (self.step_time - self.firing_period) < spike < self.step_time:
                    counter +=1
            if counter > spike_number:
                spike_neuron = ind
                spike_number = counter
        return spike_neuron, spike_number

    def end_run(self, track, title, results, name):
        #plot_activated_loc_neuron(track, results)
        plot_seen_landmarks(self.populations_m[2], "VISABLE_LANDMARKS"+title+"", self.step_time, self.firing_period, self.firing_period, name)
        plot_active_neurons(self.populations_m[2], "VISABLE_LANDMARKS"+title+"", self.step_time, self.firing_period, self.firing_period, name)
        plot_active_neurons(self.populations_m[5], "LANDMARK_X_POS_"+title+"", self.step_time, self.firing_period, self.firing_period, name)
        plot_active_neurons(self.populations_m[6], "LANDMARK_Y_POS_"+title+"", self.step_time, self.firing_period, self.firing_period, name)

        self.plot_all_results(""+title+"-"+name+"")

map = mapping_SNN()
map.train_snn()
map.test_snn()