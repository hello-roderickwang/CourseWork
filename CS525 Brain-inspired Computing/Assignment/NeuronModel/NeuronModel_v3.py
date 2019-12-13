import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pylab import *
from random import *
import math


class LIF_Neuron:
	def __init__(self, simulate_time, input_number):
		self.simulate_time = simulate_time
		self.d_time = 0.1
		self.membrane_potential = zeros(int((self.simulate_time / self.d_time) + 1))
		self.threshold = 1
		self.rest_potential = 0
		self.initial_potential = 0
		self.membrane_resistance = 1
		self.membrane_capacitance = 10
		self.time_constant = self.membrane_capacitance * self.membrane_resistance
		self.firing_rate = 10
		self.potential_difference = 0
		self.spike_number = 0
		self.not_spike_amount = 0
		self.spike_amount = 0
		self.fire_rate_array = []
		self.total_fire_rate = 0
		self.weight = []
        self.spike_time = []
		for i in range(input_number):
			random_weight = math.ceil(uniform(0, 2000) - 1000) / 1000
			self.weight.append(random_weight)

	def stimulate_neuron(self, stimulate_current):
		self.ctr = 0.0
		self.spike_amount = 0
		self.membrane_potential = zeros(int((self.simulate_time / self.d_time) + 1))
		self.potential_difference = 0
		for i in range(1, len(self.membrane_potential)):
			self.potential_difference = (-1*self.membrane_potential[i-1]+self.membrane_resistance*stimulate_current)
            #print("membrane_potential:",self.membrane_potential)
			self.membrane_potential[i] = self.membrane_potential[i-1]+self.potential_difference/self.time_constant*self.d_time
			if(self.membrane_potential[i] >= self.threshold):
                self.spike_time.append(i / len(self.membrane_potential) * simulate_time)
			    self.membrane_potential[i] = self.rest_potential
			    self.spike_amount += 1							
		return (math.ceil((self.spike_amount / self.simulate_time) * 1000)) / 1000

	def plot_neuron(self):
		simulate_time = arange(0, self.simulate_time + self.d_time, self.d_time)
		plt.plot(simulate_time, self.membrane_potential)
		plt.xlabel('Simulate Time (ms)')
		plt.ylabel('Membrane Potential (v)')
		plt.title('LIF Neuron Model')
		plt.annotate('Spike Behavior', xy = (self.spike_time[1], self.threshold), xytext = (self.spike_time[1] + 1, self.threshold + 1), arrowprops = dict(facecolor = 'red', shrink = 0.05))
		plt.axvline(x = self.spike_time[1], color = 'g')
		plt.show()
		print('Spike rate for this input is ', self.spike_amount / self.simulate_time)
        
	def calculate_fire_rate(self):
		simulate_time = arange(0, self.simulate_time + self.d_time, self.d_time)
		return self.spike_amount / self.simulate_time

if __name__ == "__main__":
	simulate_time = 20
	stimulation_number = 5
	stimulate_current = 1.2
	lif_neuron = LIF_Neuron(simulate_time, stimulation_number)
	lif_neuron.stimulate_neuron(stimulate_current)
	lif_neuron.plot_neuron()
    
	'''
	fire_rate = []
	for i in range(0, 1500):
	    lif_neuron_new = LIF_Neuron(simulate_time, stimulation_number)
	    lif_neuron_new.stimulate_neuron(i)
	    fire_rate.append(lif_neuron_new.calculate_fire_rate())
    
	plt.plot(range(0, 1500), fire_rate)
	plt.xlabel('Stimulate Current (A)')
	plt.ylabel('LIF Neuron Fire Rate')
	plt.title('Input - Fire Rate Relation')
	plt.show()
    '''

