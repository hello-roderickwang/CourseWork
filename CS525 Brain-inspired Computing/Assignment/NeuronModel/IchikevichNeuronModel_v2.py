import numpy as np
import matplotlib.pyplot as plt
from random import *

class IchikevichNeuron(self):
	def __init__(self):
		self.num_action = 800
		self.num_inhibitory = 200
		self.random_list_action = np.random.rand(self.num_action)
		self.random_list_inhibitory = np.random.rand(self.num_inhibitory)
		self.param_a = np.array([0.02*np.ones(self.num_action)], [0.02+0.08*self.random_list_inhibitory])
		self.param_b = np.array([0.2*np.ones(self.num_action)], [0.25-0.05*self.random_list_inhibitory])
		self.param_c = np.array([-65+15*self.random_list_action**2], [-65*np.ones(self.num_inhibitory)])
		self.param_d = np.array([8-6*self.random_list_action**2], [2*np.ones(self.num_inhibitory)])
		self.param_s = np.array([0.5*np.random.rand(self.num_action+self.num_inhibitory, self.num_action)], [-1*np.random.rand(self.num_action+self.num_inhibitory, self.num_inhibitory)])
		self.param_v = np.array([-65*np.ones(self.num_action+self.num_inhibitory)])
		self.param_u = 