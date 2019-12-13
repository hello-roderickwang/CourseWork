import numpy as np
import matplotlib.pyplot as plt
import math
import random

class LIF_Neuron:
	def __init__(self, input_current):
		# self.input_current = np.zeros(20)
		self.input_current = input_current
		self.dt = 1
		self.v_array = np.zeros(20)
		self.fire_array = np.zeros(20)
		self.v_threshold = 10#-10
		self.v_rest = 0#-65
		# self.v_array[0] = self.v_rest
		# self.v_array[-1] = self.v_rest
		self.r_m = 1
		self.c_m = 10
		self.tau_m = self.r_m*self.c_m
		self.neuron_status = -1 # **-1 no status**0 not fire**1 fire
		# What should the length of output_current be?
		self.output_current = np.zeros(20)
		self.current_amplify = 4.5
		self.fire_number = 0
		# self.output_current = np.zeros(len(self.input_current)*int(1/self.dt))

	def normalize(self, input_array):
		# sorted_array = np.sort(input_array, kind = 'mergesort')
		# minimum = sorted_array[0]
		# maximum = sorted_array[len(sorted_array)-1]
		# normalized_array = np.zeros(len(input_array))
		# for i in range(len(input_array)):
		# 	normalized_array[i] = (input_array[i]-minimum)/(maximum-minimum)
		# return normalized_array
		return input_array

	def start(self):
		# print('initial fire array:', self.fire_array)
		# print('initial v_array:', self.v_array)
		self.input_current = self.normalize(self.input_current)
		self.input_current = self.input_current*self.current_amplify
		self.v_array = np.zeros(len(self.input_current)*int(1/self.dt))
		self.fire_array = np.zeros(len(self.v_array))
		self.output_current = np.zeros(len(self.input_current))
		count = 0
		for i in range(1, len(self.v_array)):
			v_difference = -1*self.v_array[i-1]+self.r_m*self.input_current[math.floor(i*self.dt)]
			self.v_array[i] = self.v_array[i-1]+v_difference/self.tau_m*self.dt
			if(self.v_array[i]>=self.v_threshold):
				# print('v_array[', i, '] is:', self.v_array[i])
				# print('v_array[i-1]:', self.v_array[i-1])
				self.v_array[i] = self.v_rest
				self.fire_array[i] = 1
				count += 1
			else:
				self.v_array[i]
		self.fire_number = count
		self.output_current = self.fire_array
		# print('*NEURON* neuron fired', count, 'times')
		# # print('*NEURON* v_array:', self.v_array)
		# print('*NEURON* fire_array:', self.fire_array)
		# print('*NEURON* input_current:', self.input_current)

	def plot(self):
		plt.plot(range(0, len(self.v_array)), self.v_array)
		plt.plot(range(0, len(self.fire_array)), self.fire_array, 'o')
		# input_plot = np.zeros(len(self.input_current)*int(1/self.dt))
		# for i in range(0, len(self.input_current)*int(1/self.dt), int(1/self.dt)):
		# 	input_plot[i] = 1
		# plt.plot(range(0, len(self.input_current)*int(1/self.dt)), input_plot, 'o')
		plt.xlabel('Simulate Time in ms')
		plt.ylabel('Membrane Potential in mv')
		plt.show()

class Ichikevich_Neuron:
	def __init__(self, input_current):
		self.input_current = input_current
		self.simulate_time = 1000
		self.d_time = 0.5
		self.param_a = 0.02
		self.param_b = 0.2
		self.param_c = -65
		self.param_d = 8
		self.lapp = 10
		self.tr = np.array([200, 700])//self.d_time
		self.T = int(self.simulate_time/self.d_time)
		self.v = np.zeros(self.T)
		self.u = np.zeros(self.T)
		self.v[0] = -70
		self.u[0] = -14

	def stimulate_neuron(self):
		for i in np.arange(self.T-1):
			if i>self.tr[0] and i<self.tr[1]:
				l = self.lapp
			else:
				l = 0
			if self.v[i]<35:
				dv = (0.04*self.v[i]+5)*self.v[i]+140-self.u[i]
				self.v[i+1] = self.v[i]+(dv+l)*self.d_time
				du = self.param_a*(self.param_b*self.v[i]-self.u[i])
				self.u[i+1] = self.u[i]+self.d_time*du
			else:
				self.v[i] = 35
				self.v[i+1] = self.param_c
				self.u[i+1] = self.u[i] + self.param_d

	def plot_neuron(self):
		tvec = np.arange(0, self.simulate_time, self.d_time)
		plt.plot(tvec, self.v, 'b', label = 'Voltage Trace')
		plt.xlabel('Simulate Time')
		plt.ylabel('Membrane Potential')
		plt.title('Ichikevich Neuron Model')
		plt.show()

class Layer:
	def __init__(self, neuron_number, layer_type):
		self.neuron_number = neuron_number
		self.layer_type = 0
		if layer_type is 'input':
			self.layer_type = 1
		elif layer_type is 'hidden':
			self.layer_type = 2
		elif layer_type is 'output':
			self.layer_type = 3
		else:
			print('ERROR! undefined layer type!')
		self.layer_entity = []
		self.layer_input = np.zeros(20)
		self.layer_outpout = np.zeros(20)
		# print('*LAYER* layer_input', self.layer_input)
		self.build()

	def build(self):
		for i in range(0, self.neuron_number):
			self.layer_entity.append(LIF_Neuron(self.layer_input))
			# self.layer_entity[i] = LIF_Neuron(self.layer_input)

class Synapse:
	def __init__(self, pre_neuron, post_neuron):
		self.pre_neuron = pre_neuron
		self.post_neuron = post_neuron
		self.weight = random.randrange(0, 5)
		self.wire()
		self.calculate_current()

	def set_weight(self, weight):
		self.weight = weight

	def calculate_current(self):
		# print('*CALCULATE CURRENT* post_neuron', self.post_neuron)		
		# print('*CALCULATE CURRENT* post_neuron.input_current:', self.post_neuron.input_current)
		# print('*CALCULATE CURRENT* pre_neuron.output_current:', self.pre_neuron.output_current)
		# # self.post_neuron.input_current = np.add(self.post_neuron.input_current, np.dot(self.pre_neuron.output_current, self.weight))
		if self.weight<0:
			self.weight = 0
		elif self.weight>5:
			self.weight = 5
		self.post_neuron.input_current = np.add(self.post_neuron.input_current, self.pre_neuron.output_current*self.weight)
		# print('*CALCULATE CURRENT* weight:', self.weight)

	# def get_current(self, extra_current):
	# 	for i in range(0, len(extra_current)):
	# 		self.post_neuron.input_current[i] = self.post_neuron.input_current[i]+extra_current[i]

	def wire(self):
		# self.post_neuron.input_current = self.pre_neuron.output_current
		self.post_neuron.input_current = self.pre_neuron.fire_array


class SNN:
	def __init__(self, layer_number, input_number, output_number):
		self.layer_number = layer_number
		self.input_number = input_number
		self.output_number = output_number
		self.network_entity = []
		self.input_layer = Layer(self.input_number, 'input')
		self.output_layer = Layer(self.output_number, 'output')
		# self.network_entity[0] = self.input_layer
		# self.network_entity[self.layer_number-1] = self.output_layer
		self.synapse_entity = [[[0 for post in range(8)]for pre in range(8)]for net in range(self.layer_number-1)]
		self.learning_rate = 1
		self.decode_threshold = 0.5
		self.current_length = 20
		self.layers = []

		# self.neuronClass = neuron_class
		# self.learning = learning_method
		# self.setup(num_input, hidden_layers, num_output)

		# self.hidden_layer_neuron_number = 0
		# self.current_amplify = 50

	def set_input(self, input_current_x, input_current_y):
		# # self.network_entity[0].layer_input = self.generate_current(input_current, self.current_length)
		# # print('*SET_INPUT* layer_input:', self.network_entity[0].layer_input)
		# print('*SET_INPUT* current_amplify:', self.current_amplify)
		# print('******************************************')
		# print('*SET_INPUT* input_current_x:', input_current_x)
		# print('*SET_INPUT* input_current_y:', input_current_y)
		# print('******************************************')
		self.network_entity[0].layer_entity[0].input_current = self.generate_current(input_current_x, self.current_length)
		self.network_entity[0].layer_entity[1].input_current = self.generate_current(input_current_y, self.current_length)

	def get_output(self):
		return self.network_entity[self.layer_number-1].layer_outpout

	def build_hidden(self, neuron_number):
		for i in range(0, len(neuron_number)):
			self.network_entity[i+1] = Layer(neuron_number[i], 'hidden')

	def build(self, hidden_layer_neuron_number):
		self.hidden_layer_neuron_number = hidden_layer_neuron_number
		if len(hidden_layer_neuron_number) is not self.layer_number-2:
			print('Hidden layer number wrong!')
		self.network_entity.append(Layer(self.input_number, 'input'))
		for i in range(0, self.layer_number-2):
			self.network_entity.append(Layer(hidden_layer_neuron_number[i], 'hidden'))
		self.network_entity.append(Layer(self.output_number, 'output'))
		# print('*NETWORK BUILD* network_entity:', self.network_entity)
		# for i in range(len(self.network_entity)):
		# 	for j in range(self.network_entity[i].neuron_number):
		# 		print('*NETWORK BUILD* network_entity[i].layer_entity[j].input_current', self.network_entity[i].layer_entity[j].input_current)
		# 		print('*NETWORK BUILD* network_entity[i].layer_entity[j].output_current', self.network_entity[i].layer_entity[j].output_current)


	# def decode_fire_array(self, input_current):
	# 	if np.sum(input_current)/len(input_current)>=self.decode_threshold:
	# 		return 1
	# 	else:
	# 		return 0

	# def get_random(self, low, high):
	# 	return random.randint(low, high)

	def generate_current(self, current_type, current_length):
		# print('*GENERATE_CURRENT* current_type:', current_type)
		pulse_number = 0
		# pulse_list = np.zeros(current_length)
		if current_type is 0:
			pulse_number = random.randint(0, math.floor(current_length/2))
		elif current_type is 1:
			pulse_number = random.randint(math.ceil(current_length/2), current_length)
		else:
			print('*GENERATE_CURRENT* current_type is wrong!')
		pulse_list = np.ones(pulse_number, dtype = int)
		pulse_list = np.append(pulse_list, np.zeros(current_length-pulse_number))
		np.random.shuffle(pulse_list)
		# print('*GENERATE_CURRENT* pulse_number:', pulse_number)
		# print('*GENERATE_CURRENT* pulse_list:', pulse_list)
		return pulse_list
		# if current_type is 0:
		# 	pulse_number = random.randint(0, math.floor(current_length/2))
		# 	for n in range(0, pulse_number):
		# 		rand = random.randint(0, current_length)
		# 		if len(rand_list)>0:
		# 			for m in range(0, n):
		# 				if rand is rand_list[m]:
		# 					rand = random.randint(0, current_length)
		# 					m = 0
		# 		np.append(rand_list, rand)
		# 		pulse_list[rand] = 1
		# elif current_type is 1:
		# 	pulse_number = random.randint(math.ceil(current_length/2), current_length)
		# 	for n in range(0, pulse_number):
		# 		rand = random.randint(0, current_length)
		# 		if len(rand_list)>0:
		# 			for m in range(0, n):
		# 				if rand is rand_list[m]:
		# 					rand = random.randint(0, current_length)
		# 					m = 0
		# 		np.append(rand_list, rand)
		# 		pulse_list[rand] = 1
		# print('*GENERATE_CURRENT* pulse_list:', pulse_list)
		# print('*GENERATE_CURRENT* rand_list:', rand_list)

	def interconnect(self):
		for net in range(0, self.layer_number-1):
			for pre in range(0, self.network_entity[net].neuron_number):
				for post in range(0, self.network_entity[net+1].neuron_number):
					# print('net:', net)
					# print('pre', pre)
					# print('post', post)
					# print('synapse_entity:', self.synapse_entity)
					self.synapse_entity[net][pre][post] = Synapse(self.network_entity[net].layer_entity[pre], self.network_entity[net+1].layer_entity[post])

	def teaching_agent(self, target_neuron, target_status):
		if target_status is 0:
			# target_neuron.input_current = 1/5*target_neuron.input_current
			target_neuron.input_current = np.zeros(len(target_neuron.input_current))
		elif target_status is 1:
			# target_neuron.input_current = 5*target_neuron.input_current
			target_neuron.input_current = np.ones(len(target_neuron.input_current))*10
		else:
			print('ERROR! wrong target_status!')

	# def hebbian_learning(self):
	# 	for net in range(1, len(self.network_entity)):
	# 		for post in range(0, len(self.network_entity[net].layer_entity)):
	# 			for pre in range(0, len(self.network_entity[net-1].layer_entity)):
	# 				for i in range(0, len(self.network_entity[net].layer_entity[post].output_current))
	# 					weight_diff = self.learning_rate*self.network_entity[net].layer_entity[post].output_current[i]*\
	# 					(self.network_entity[net-1].layer_entity[pre].output_current[i]-self.network_entity[net].layer_entity[post].output_current[i]*\
	# 					self.synapse_entity[net-1, pre, post])
	# 					self.synapse_entity[net-1, pre, post] = self.synapse_entity[net-1, pre, post]+weight_diff

	def stdp_setup(self, num_input, hidden_layers, num_output):
		self.setup_layer(num_input)
		self.setup_hidden(hidden_layers)
		self.setup_layer(num_output)

	def stdp_setup_layer(self, num_neurons):
		layer_neurons = np.array([])
		for x in range(num_neurons):
			input_weights = len(self.layers[-1]) if len(self.layers) > 0 else num_neurons
			layer_neurons = np.append(layer_neurons, self.neuronClass(input_weights))
		self.layers.append(layer_neurons)

	def stdp_setup_hidden(self, hidden_layers):
		if type(hidden_layers) is int:
			self.setup_layer(hidden_layers)
		else:
			for layer in hidden_layers:
				self.setup_layer(layer)

	def stdp_adjust_weights(self):
		self.learning.update(self.layers)

	def stdp_solve(self, input):
		previous_layer = np.array(input)
		for (i, layer) in enumerate(self.layers):
			new_previous_layer = np.array([])
			for neuron in layer:
				new_previous_layer = np.append(new_previous_layer, neuron.solve(previous_layer))
			previous_layer = new_previous_layer
		self.adjust_weights()
		return previous_layer

	def normalize(self, input_array):
		# sorted_array = np.sort(input_array, kind = 'mergesort')
		# minimum = sorted_array[0]
		# maximum = sorted_array[len(sorted_array)-1]
		# normalized_array = np.zeros(len(input_array))
		# for i in range(len(input_array)):
		# 	normalized_array[i] = (input_array[i]-minimum)/(maximum-minimum)
		# return normalized_array
		return input_array

	def hebbian_learning(self, net):
		for post in range(0, len(self.network_entity[net].layer_entity)):
			# print('*HEBBIAN_LEARNING* llen(self.network_entity[net].layer_entity)):', len(self.network_entity[net].layer_entity))
			for pre in range(0, len(self.network_entity[net-1].layer_entity)):
				# # print('*HEBBIAN_LEARNING* llen(self.network_entity[net].layer_entity)):', len(self.network_entity[net].layer_entity))
				# for i in range(0, len(self.network_entity[net].layer_entity[post].output_current)):
				# 	# print('*HEBBIAN_LEARNING* synapse_entity[net][pre][post]:', self.synapse_entity[net-1][pre][post])
				# 	# print('*HEBBIAN_LEARNING* len(synapse_entity[:][:][:]):', len(self.synapse_entity[:][:][:]))
				# 	if self.synapse_entity[net-1][pre][post] is not 0:
				# 		weight_diff = self.learning_rate*self.network_entity[net].layer_entity[post].output_current[i]*\
				# 		(self.network_entity[net-1].layer_entity[pre].output_current[i]-self.network_entity[net].layer_entity[post].output_current[i]*\
				# 		self.synapse_entity[net-1][pre][post].weight)
				# 		self.synapse_entity[net-1][pre][post].weight = self.synapse_entity[net-1][pre][post].weight+weight_diff
				if self.synapse_entity[net-1][pre][post] is not 0:
					weight_diff = self.learning_rate*self.network_entity[net].layer_entity[post].fire_number*\
					(self.network_entity[net-1].layer_entity[pre].fire_number-self.network_entity[net].layer_entity[post].fire_number*\
					self.synapse_entity[net-1][pre][post].weight)
					self.synapse_entity[net-1][pre][post].weight = self.synapse_entity[net-1][pre][post].weight+weight_diff

	def start_training(self, target_output_status_1, target_output_status_2):
		for net in range(0, self.layer_number):
			for layer in range(0, self.network_entity[net].neuron_number):
				self.network_entity[net].layer_entity[layer].start()
			for i in range(4):
				for j in range(4):
					if net <= self.layer_number-2:
						if self.synapse_entity[net][i][j] is not 0:
							self.synapse_entity[net][i][j].calculate_current()
			if net is self.layer_number-2:
				self.teaching_agent(self.network_entity[net].layer_entity[0], target_output_status_1)
				self.teaching_agent(self.network_entity[net].layer_entity[1], target_output_status_2)
		for net in range(0, self.layer_number):
			self.hebbian_learning(net)
		for net in range(0, self.layer_number-1):
			for post in range(len(self.network_entity[net+1].layer_entity)):
				weight_array = np.zeros(self.network_entity[net].neuron_number)
				for pre in range(len(self.network_entity[net].layer_entity)):
					if self.synapse_entity[net][pre][post] is not 0:
						weight_array[pre] = self.synapse_entity[net][pre][post].weight
						# print('*START_TRAINING* weight_array[pre]:', weight_array[pre])
						# print('*START_TRAINING* synapse_entity[net-1][pre][post].weight:', self.synapse_entity[net-1][pre][post].weight)
				normalized_array = self.normalize(weight_array)
				for pre in range(len(self.network_entity[net].layer_entity)):
					if self.synapse_entity[net][pre][post] is not 0:
						self.synapse_entity[net][pre][post].weight = normalized_array[pre]


	def test(self, test_x, test_y):
		self.set_input(test_x, test_y)
		for net in range(self.layer_number):
			for layer in range(self.network_entity[net].neuron_number):
				self.network_entity[net].layer_entity[layer].start()
			for i in range(4):
				for j in range(4):
					if net <= self.layer_number-2:
						if self.synapse_entity[net][i][j] is not 0:
							self.synapse_entity[net][i][j].calculate_current()
		print('*TEST* output_current', self.network_entity[self.layer_number-1].layer_entity[0].output_current)
		return self.network_entity[self.layer_number-1].layer_entity[0].fire_number, self.network_entity[self.layer_number-1].layer_entity[1].fire_number


if __name__ == '__main__':
	input_current = [1,1,1,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1]
	input_current = np.zeros(300)
	for i in range(0, len(input_current)):
		input_current[i] = random.randrange(0, 2000)/1000
	print('input length:', len(input_current))
	print('input current:', input_current)
	lif = LIF_Neuron(input_current)
	lif.start()
	lif.plot()
	print('final fire array:', lif.fire_array)
	print('final fire array:', lif.v_array)

	# current_threshold = 5000
	xor_input = [[0, 0], [1, 0], [0, 1], [1, 1]]
	hebbian_learning = SNN(3, 2, 2)
	training_times = 10
	hidden_layer_neuron_number = 4
	hebbian_learning.build([hidden_layer_neuron_number])
	hebbian_learning.interconnect()
	for n in range(training_times):
		for i in range(0, 4):
			hebbian_learning.set_input(xor_input[i][0], xor_input[i][1])
			# for n in range(0, len(xor_input[i])):
			# 	print('i:', i)
			# 	print('n:' ,n)
			# 	print('xor_input[i, n]:', xor_input[i][n])
			# 	hebbian_learning.set_input(xor_input[i][n])
			# hebbian_learning.set_input(xor_input[i, :])
			if i is 0 or 3:
				hebbian_learning.start_training(0, 1)
				print('@@@@@@@')
				print('output fire_number:', hebbian_learning.network_entity[2].layer_entity[0].fire_number)
				print('output fire_number:', hebbian_learning.network_entity[2].layer_entity[1].fire_number)
				print('@@@@@@@')
				print('-----------------------------------------------')
			else:
				print('@@@@@@@')
				print('output fire_number:', hebbian_learning.network_entity[2].layer_entity[0].fire_number)
				print('output fire_number:', hebbian_learning.network_entity[2].layer_entity[1].fire_number)
				print('@@@@@@@')
				hebbian_learning.start_training(1, 0)
				print('-----------------------------------------------')
	# for i in range(0, 4):
	# 	hebbian_learning.build([4])
	# 	hebbian_learning.interconnect()
	# 	hebbian_learning.set_input(xor_input[i][0], xor_input[i][1])
	# 	# for n in range(0, len(xor_input[i])):
	# 	# 	print('i:', i)
	# 	# 	print('n:' ,n)
	# 	# 	print('xor_input[i, n]:', xor_input[i][n])
	# 	# 	hebbian_learning.set_input(xor_input[i][n])
	# 	# hebbian_learning.set_input(xor_input[i, :])
	# 	if i is 0 or 3:
	# 		hebbian_learning.start_training(0)
	# 		print('@@@@@@@')
	# 		print('output fire_number:', hebbian_learning.network_entity[2].layer_entity[0].fire_number)
	# 		print('@@@@@@@')
	# 		print('-----------------------------------------------')
	# 	else:
	# 		print('@@@@@@@')
	# 		print('output fire_number:', hebbian_learning.network_entity[2].layer_entity[0].fire_number)
	# 		print('@@@@@@@')
	# 		hebbian_learning.start_training(1)
	# 		print('-----------------------------------------------')
	print('input_1 input_current:', hebbian_learning.network_entity[0].layer_entity[0].input_current)
	print('input_1 fire_array:', hebbian_learning.network_entity[0].layer_entity[0].fire_array)
	print('input_2 input_current:', hebbian_learning.network_entity[0].layer_entity[1].input_current)
	print('input_2 fire_array:', hebbian_learning.network_entity[0].layer_entity[1].fire_array)


	print('output_1 input_current:', hebbian_learning.network_entity[2].layer_entity[0].input_current)
	print('output_1 fire_array:', hebbian_learning.network_entity[2].layer_entity[0].fire_array)
	print('output_2 input_current:', hebbian_learning.network_entity[2].layer_entity[1].input_current)
	print('output_2 fire_array:', hebbian_learning.network_entity[2].layer_entity[1].fire_array)

	print('network_entity:', hebbian_learning.network_entity)
	print('size of network_entity:', len(hebbian_learning.network_entity))
	# hebbian_learning.network_entity[2].layer_entity[0].plot()
	print('!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()')
	print('-----------------------test-----------------------')
	print('input: 0 0\nfire_number:', hebbian_learning.test(0, 0))
	print('input: 0 1\nfire_number:', hebbian_learning.test(0, 1))
	print('input: 1 0\nfire_number:', hebbian_learning.test(1, 0))
	print('input: 1 1\nfire_number:', hebbian_learning.test(1, 1))

	print('---------------EVERY INPUT OUTPUT-----------------')
	for net in range(hebbian_learning.layer_number):
		print('Layer:', net)
		for layer in range(hebbian_learning.network_entity[net].neuron_number):
			print('^^^Neuron No.', layer, '^^^')
			print('Input:', hebbian_learning.network_entity[net].layer_entity[layer].input_current)
			print('Output:', hebbian_learning.network_entity[net].layer_entity[layer].output_current)
		print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


	# 自己写一个归一化函数
	# 将post_neuron的所有输入weight进行归一化处理

	# using two output neurons
	# exp建立一个array对应每一个时间应有的下降数值 然后后面直接减去此数组中对应的数值
