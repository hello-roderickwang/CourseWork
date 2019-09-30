#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-04-20 16:05:54
# @Author  : Xuenan(Roderick) Wang
# @email   : roderick_wang@outlook.com
# @Link    : https://github.com/hello-roderickwang

import nengo
from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves
from nengo.utils.ipython import hide_input
from nengo.utils.matplotlib import rasterplot
from nengo.processes import WhiteSignal
import numpy as np
import matplotlib.pyplot as plt

model = nengo.Network()

def product(x):
	return x[0]*x[1]

with model:
	# nengo.Ensemble is a group of neurons that represents information in the form of real value numbers.
	# by default, Nengo uses LIF neurons
	my_ensemble = nengo.Ensemble(n_neurons=40, dimensions=1)
	# provide input to this ensemble
	# my_node emits the number 0.5
	my_node = nengo.Node(output=0.5)
	sin_node = nengo.Node(output=np.sin)
	nengo.Connection(my_node, my_ensemble)
	two_d_ensemble = nengo.Ensemble(n_neurons=80, dimensions=2)
	nengo.Connection(sin_node, two_d_ensemble[0])
	nengo.Connection(my_ensemble, two_d_ensemble[1])
	square = nengo.Ensemble(n_neurons=40, dimensions=1)
	nengo.Connection(my_ensemble, square, function=np.square)
	product_ensemble = nengo.Ensemble(n_neurons=40, dimensions=1)
	nengo.Connection(two_d_ensemble, product_ensemble, function=product)
	two_d_probe = nengo.Probe(two_d_ensemble, synapse=0.01)
	product_probe = nengo.Probe(product_ensemble, synapse=0.01)

def aligned(n_neurons, radius=0.9):
	intercepts = np.linspace(-radius, radius, n_neurons)
	encoders = np.tile([[1], [-1]], (n_neurons // 2, 1))
	intercepts *= encoders[:, 0]
	return intercepts, encoders

model2 = nengo.Network(label="NEF Summary")
intercepts, encoders = aligned(8)
with model2:
	input = nengo.Node(lambda t:t*2-1)
	input_probe = nengo.Probe(input)
	# figure 2-5
	# A = nengo.Ensemble(8, dimensions=1, intercepts=intercepts, max_rates=Uniform(80, 100), encoders=encoders)
	# figure 6
	A = nengo.Ensemble(30, dimensions=1, max_rates=Uniform(80, 100))
	nengo.Connection(input, A)
	# figure 2-3 6
	A_spikes = nengo.Probe(A.neurons)
	# figure 4-5
	# A_spikes = nengo.Probe(A.neurons, synapse=0.01)
	A_probe = nengo.Probe(A, synapse=0.01)

if __name__ == '__main__':
	sim = nengo.Simulator(model)
	sim.run(5.0)
	print(sim.data[product_probe][-10:])

	with nengo.Simulator(model2) as sim:
		sim.run(1.0)

	plt.figure(1)
	plt.plot(sim.trange(), sim.data[input_probe], lw=2)
	plt.title("Input signal")
	plt.xlabel("Time (s)")
	plt.xlim(0, 1)

	with nengo.Simulator(model2) as sim:
		eval_points, activities = tuning_curves(A, sim)

	plt.figure(2)
	plt.plot(eval_points, activities, lw=2)
	plt.xlabel("Input Signal")
	plt.ylabel("Firing rate (Hz)")

	with nengo.Simulator(model2) as sim:
		sim.run(1)

	plt.figure(3)
	ax = plt.subplot(1, 1, 1)
	rasterplot(sim.trange(), sim.data[A_spikes], ax)
	ax.set_xlim(0, 1)
	ax.set_ylabel('Neuron')
	ax.set_xlabel('Time (s)')

	with nengo.Simulator(model2) as sim:
		sim.run(1)

	scale = 180
	plt.figure(4)
	for i in range(A.n_neurons):
		plt.plot(sim.trange(), sim.data[A_spikes][:, i] - i * scale)
	plt.xlim(0, 1)
	plt.ylim(scale * (-A.n_neurons + 1), scale)
	plt.ylabel("Neuron")
	plt.yticks(
		np.arange(scale / 1.8, (-A.n_neurons + 1) * scale, -scale),
		np.arange(A.n_neurons))

	with nengo.Simulator(model2) as sim:
		sim.run(1)

	plt.figure(5)
	plt.plot(sim.trange(), sim.data[input_probe], label="Input signal")
	plt.plot(sim.trange(), sim.data[A_probe], label="Decoded estimate")
	plt.legend(loc="best")
	plt.xlim(0, 1)

	with nengo.Simulator(model2) as sim:
		sim.run(1)

	plt.figure(6, figsize=(15, 3.5))
	plt.subplot(1, 3, 1)
	eval_points, activities = tuning_curves(A, sim)
	plt.plot(eval_points, activities, lw=2)
	plt.xlabel("Input signal")
	plt.ylabel("Firing rate (Hz)")

	ax = plt.subplot(1, 3, 2)
	rasterplot(sim.trange(), sim.data[A_spikes], ax)
	plt.xlim(0, 1)
	plt.xlabel("Time (s)")
	plt.ylabel("Neuron")

	plt.subplot(1, 3, 3)
	plt.plot(sim.trange(), sim.data[input_probe], label="Input signal")
	plt.plot(sim.trange(), sim.data[A_probe], label="Decoded estimate")
	plt.legend(loc="best")
	plt.xlabel("Time (s)")
	plt.xlim(0, 1)

	plt.show()


