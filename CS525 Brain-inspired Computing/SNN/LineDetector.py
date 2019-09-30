import numpy as np
import matplotlib.pyplot as plt
import math
import random

class LIF_Neuron:
	def __init__(self, input_current):
		self.input_current = np.zeros(20)
		self.dt = 1
		self.v_array = np.zeros(20)
		self.fire_array = np.zeros(20)
		self.v_threshold = 1#-10
		self.v_rest = 0#-65
		self.r_m = 1
		self.c_m = 10
		self.tau_m = self.r_m*self.c_m
		self.neuron_status = -1 # **-1 no status**0 not fire**1 fire
		self.output_current = np.zeros(20)
		self.current_amplify = 3
		self.fire_number = 0

	def normalize(self, input_array):
		sorted_array = np.sort(input_array, kind = 'mergesort')
		minimum = sorted_array[0]
		maximum = sorted_array[len(sorted_array)-1]
		normalized_array = np.zeros(len(input_array))
		for i in range(len(input_array)):
			normalized_array[i] = (input_array[i]-minimum)/(maximum-minimum)
		return normalized_array

	def start(self):
		count = 0
		self.input_current = self.input_current*self.current_amplify
		for i in range(1, len(self.v_array)):
			v_difference = -1*self.v_array[i-1]+self.r_m*self.input_current[math.floor(i*self.dt)]
			self.v_array[i] = self.v_array[i-1]+v_difference/self.tau_m*self.dt
			if(self.v_array[i]>=self.v_threshold):
				self.v_array[i] = self.v_rest
				self.fire_array[i] = 1
				count += 1
		self.fire_number = count
		self.output_current = self.fire_array
		# print('$$$$ v_array:', self.v_array)

	def plot(self):
		plt.plot(range(0, len(self.v_array)), self.v_array)
		plt.plot(range(0, len(self.fire_array)), self.fire_array, 'o')
		plt.xlabel('Simulate Time in ms')
		plt.ylabel('Membrane Potential in mv')
		plt.show()

class LineDetector:
	def __init__(self, target_matrix):
		self.target_matrix = target_matrix
		self.input_matrix = [[0 for i in range(10)]for j in range(10)]
		for i in range(10):
			for j in range(10):
				self.input_matrix[i][j] = LIF_Neuron(self.generate_current(int(self.target_matrix[i][j]), 20))
		self.output = LIF_Neuron(np.zeros(20))
		self.weight_matrix = np.random.random((10, 10))
		self.learning_rate = 1

	def set_input(self, matrix):
		for i in range(10):
			for j in range(10):
				self.input_matrix[i][j].input_current = self.generate_current(int(matrix[i][j]), 20)

	def generate_current(self, current_type, current_length):
		current_type = int(current_type)
		# print('*GENERATE_CURRENT* current_type:', current_type)
		pulse_number = 0
		# pulse_list = np.zeros(current_length)
		if current_type is 0:
			pulse_number = random.randint(0, math.floor(current_length/2))
		else:
		# else current_type is 1:
			pulse_number = random.randint(math.ceil(current_length/2), current_length)
		# else:
		# 	print('*GENERATE_CURRENT* current_type is wrong! current_type:', current_type)
		pulse_list = np.ones(pulse_number, dtype = int)
		pulse_list = np.append(pulse_list, np.zeros(current_length-pulse_number))
		np.random.shuffle(pulse_list)
		# print('*GENERATE_CURRENT* pulse_number:', pulse_number)
		# print('*GENERATE_CURRENT* pulse_list:', pulse_list)
		return pulse_list

	def start_training_positive(self, positive_matrix):
		self.set_input(positive_matrix)
		for i in range(10):
			for j in range(10):
				self.input_matrix[i][j].start()
				for n in range(20):
					self.output.input_current[n] = self.input_matrix[i][j].output_current[n]*self.weight_matrix[i][j]+self.output.input_current[n]
		self.output.start()
		self.output.output_current = np.ones(20, dtype = int)*500
		self.output.fire_array = self.output.output_current
		for i in range(10):
			for j in range(10):
				for n in range(20):
					diff_weight = self.learning_rate*self.output.output_current[n]*(self.input_matrix[i][j].output_current[n]-self.output.output_current[n]*self.weight_matrix[i][j])
					self.weight_matrix[i][j] = self.weight_matrix[i][j]+diff_weight
		print('positive weight_matrix:', self.weight_matrix)

	def start_training_negative(self, negative_matrix):
		self.set_input(negative_matrix)
		for i in range(10):
			for j in range(10):
				self.input_matrix[i][j].start()
				for n in range(20):
					self.output.input_current[n] = self.input_matrix[i][j].output_current[n]*self.weight_matrix[i][j]+self.output.input_current[n]
		self.output.start()
		self.output.output_current = np.zeros(20, dtype = int)*500
		self.output.fire_array = self.output.output_current
		for i in range(10):
			for j in range(10):
				for n in range(20):
					diff_weight = self.learning_rate*self.output.output_current[n]*(self.input_matrix[i][j].output_current[n]-self.output.output_current[n]*self.weight_matrix[i][j])
					self.weight_matrix[i][j] = self.weight_matrix[i][j]+diff_weight
		print('negative weight_matrix:', self.weight_matrix)

	def test(self, test_matrix):
		self.weight_matrix = np.dot(self.target_matrix, 1)
		for i in range(10):
			for j in range(10):
				self.input_matrix[i][j].input_current = self.generate_current(int(test_matrix[i][j]), 20)
				# print('self.input_matrix[',i,'][',j,'].input_current:', self.input_matrix[i][j].input_current)
				self.input_matrix[i][j].start()
				for n in range(20):
					# # self.output.input_current[n] = (self.input_matrix[i][j].output_current[n]*self.weight_matrix[i][j]+self.output.input_current[n])*500
					# a = self.weight_matrix[i][j]
					# b = self.input_matrix[i][j].output_current[n]
					# self.output.input_current[n]
					self.output.input_current[n] = self.weight_matrix[i][j]*self.input_matrix[i][j].output_current[n]+self.output.input_current[n]
					# self.output.input_current[n] = np.dot(self.weight_matrix[i][j], self.input_matrix[i][j].output_current[n])
					# if i is 1 and j is 5:
						# print('### output.input_current[',n,']:',self.output.input_current[n])
						# print('### weight_matrix[',i,'][',j,']:',self.weight_matrix[i][j])
						# print('### input_matrix.output_current[',n,']:',self.input_matrix[i][j].output_current[n])
						# print('### input_matrix.fire_array:', self.input_matrix[i][j].fire_array)

		# print('weight_matrix:\n', self.weight_matrix)
		# print('output.input_current:', self.output.input_current)
		self.output.start()
		# print('output.v_array:', self.output.v_array)
		count = 0
		for i in range(20):
			if self.output.fire_array[i]>0:
				count += 1
		if count >= 10:
			return 1
		else:
			return 0

	def rotate(self, matrix, angle):
		target = np.zeros((10, 10))
		for i in range(10):
			for j in range(10):
				if matrix[i][j] == 1:
					target_x = (j-5)*math.cos(angle)-(i-5)*math.sin(angle)+5
					target_y = (j-5)*math.sin(angle)+(i-5)*math.cos(angle)+5
					# print('target_x:', target_x, '  target_y:', target_y)
					target[int(target_y)][int(target_x)] = 1
		return target

if __name__ == '__main__':
	target_matrix = np.zeros((10, 10))
	for i in range(10):
		for j in range(10):
			if j is 5:
				target_matrix[i][j] = 1
	print('target_matrix:\n', target_matrix)
	print('*****************************************')
	detector = LineDetector(target_matrix)
	result = np.zeros(18)
	for n in range(18):
		result[n] = 1-detector.test(detector.rotate(target_matrix, 20*n))
		print('rotate matrix:\n', detector.rotate(target_matrix, 20*n))
		# print('TEST output.fire_array:', detector.output.fire_array)
	print('result:', result)






