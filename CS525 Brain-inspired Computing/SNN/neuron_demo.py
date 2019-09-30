import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from random import *
import math

class Neuron:
	def __init__(self, neuron_type, stimulate_current):
		self.neuron_type = neuron_type
		print('neuron_type:',self.neuron_type)
		self.stimulate_current = stimulate_current
		self.dt = 0.1
		# self.v_array = np.zeros(len(self.stimulate_current)*int(1/self.dt) + 1)
		self.v_array = np.zeros(len(self.stimulate_current)*int(1/self.dt))
		self.fire_array = np.zeros(len(self.v_array)-1)
		self.v_threshold = -10
		self.v_rest = -65
		self.v_array[0] = self.v_rest

	# LIF neuron defination
	def leaky_integrate_fire(self):
		print('This is LIF neuron model!')
		r_m = 1
		c_m = 10
		tau_m = r_m * c_m
		self.v_array[0] = self.v_rest
		for i in range(0, len(self.v_array)):
			# print('i:',i)
			v_difference = -1*self.v_array[i-1]+r_m*self.stimulate_current[math.floor(i*self.dt)]
			self.v_array[i] = self.v_array[i-1]+v_difference/tau_m*self.dt
			if(self.v_array[i]>=self.v_threshold):
				self.v_array[i] = self.v_rest
				self.fire_array[i] = 1

	# Ichikevich neuron defination
	def ichikevich(self):
		print('This is Ichikevich neuron model!')
		a = 0.02
		b = 0.2
		c = -65
		d = 8
		lapp = 10
		u_array = np.zeros(len(self.v_array))
		tr = np.array([200, 700])//self.dt
		u_array[0] = -14
		for i in np.arange(len(self.v_array)-1):
			if i>tr[0] and i<tr[1]:
				l = lapp
			else:
				l = 0
			if self.v_array[i]<self.v_threshold:
				dv = (0.04*self.v_array[i]+5)*self.v_array[i]+140-u_array[i]
				self.v_array[i+1] = self.v_array[i]+(dv+l)*self.dt
				du = a*(b*self.v_array[i]-u_array[i])
				u_array[i+1] = u_array[i]+du*self.dt
			else:
				self.v_array[i] = self.v_threshold
				self.fire_array[i] = 1
				self.v_array[i+1] = c
				u_array[i+1] = u_array[i]+d

	# Hodgkin-Huxley neuron defination
	c_m = 1.0
	g_Na = 120.0
	g_K = 36.0
	g_L = 0.3
	e_Na = 50.0
	e_K = -77.0
	e_L = -54.387

	def alpha_m(self, V):
		return 0.1*(V+40.0)/(1.0 - math.exp(-(V+40.0) / 10.0))

	def beta_m(self, V):
		return 4.0*math.exp(-(V+65.0) / 18.0)

	def alpha_h(self, V):
		return 0.07*math.exp(-(V+65.0) / 20.0)

	def beta_h(self, V):
		return 1.0/(1.0 + math.exp(-(V+35.0) / 10.0))

	def alpha_n(self, V):
		return 0.01*(V+55.0)/(1.0 - math.exp(-(V+55.0) / 10.0))

	def beta_n(self, V):
		return 0.125*math.exp(-(V+65) / 80.0)

	def I_Na(self, V, m, h):
		return g_Na * m**3 * h * (V - e_Na)

	def I_K(self, V, n):
		return g_K  * n**4 * (V - e_K)

	def I_L(self, V):
		return g_L * (V - e_L)

	def I_inj(self, t):
		return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

	@staticmethod
	def dALLdt(X, t, self):
		V, m, h, n = X

		dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / c_m
		dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
		dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
		dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
		return dVdt, dmdt, dhdt, dndt

	def hodgkin_huxley(self):
		print('This is Hodgkin-Huxley neuron model!')
		X = integrate.odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], np.arange(0.0, len(self.stimulate_current), self.dt), args=(self,))
		for i in range(0, len(self.v_array)-1):
			self.v_array[i] = X[i, 0]
			if self.v_array[i]>=self.v_threshold:
				self.fire_array[i] = 1
		# V = X[:,0]
		# m = X[:,1]
		# h = X[:,2]
		# n = X[:,3]
		# ina = self.I_Na(V, m, h)
		# ik = self.I_K(V, n)
		# il = self.I_L(V)

	def run(self):
		if self.neuron_type is 'lif':
			print('go into lif')
			self.leaky_integrate_fire()
		elif self.neuron_type is 'ichikevich':
			print('go into ichikevich')
			self.ichikevich()
		elif self.neuron_type is 'hodgkin_huxley':
			print('go into hodgkin_huxley')
			self.hodgkin_huxley()
		else:
			print('neuron type is not defined!')

	def plot_neuron(self):
		plt.plot(range(0, len(self.v_array)), self.v_array)
		plt.plot(range(0, len(self.fire_array)), self.fire_array, 'o')
		plt.plot(range(0, len(self.stimulate_current)), self.stimulate_current, 'o')
		plt.xlabel('Simulate Time in ms')
		plt.ylabel('Membrane Potential in mv')
		plt.show()

class SNN:
	def __init__(self):


if __name__ == '__main__':
	stimulate_current = [1,1,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,1]
	lif_neuron = Neuron('lif', stimulate_current)
	lif_neuron = Neuron('lif', np.dot(stimulate_current*1000, 10))
	lif_neuron.run()
	lif_neuron.plot_neuron()
	# print('????????????????')
	# ichikevich_neuron = Neuron('ichikevich', np.dot(stimulate_current*5, 50))
	# ichikevich_neuron.run()
	# ichikevich_neuron.plot_neuron()
	# print('fire_array:', ichikevich_neuron.fire_array)














