from pylab import *
import numpy as np
import matplotlib.pyplot as plt

class Ichikevich_Neuron:
	def __init__(self):
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

if __name__ == "__main__":
    #Ichikevich Neuron Model
	ichikevich_neuron = Ichikevich_Neuron()
	ichikevich_neuron.stimulate_neuron()
	ichikevich_neuron.plot_neuron()