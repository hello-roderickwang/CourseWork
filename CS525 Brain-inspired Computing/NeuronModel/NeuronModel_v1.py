# -*- coding: utf-8 -*-
"""
Copyright <2019> <Xuenan(Roderick) Wang>

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import matplotlib.pyplot as plt
import numpy as np



class Unit:
    def __init___(self, )

dv/dt = ( -(v-v_rest) + membrane_resistance * input_current(t,i) ) / membrane_time_scale : volt (unless refractory)

class LITModel:
    def __init__(self, u_rest, u_threshold, inject_current, leaky_rate, time):
        """
        initiate class NeuronModel

        :param u_now: voltage of membrane potential tight now
        :param u_rest: rest voltage of membrane potential
        :param u_threshold: threshold voltage for a spike
        """
        self.u_rest = u_rest
        self.u_threshold = u_threshold
        self.inject_current = inject_current
        self.leaky_rate = leaky_rate
        self.u_now = self.u_rest
        self.stimulate = []
        self.inject_charge = 0
        self.spike_status= 0
        self.time = time

    def stimulateNeuron(self):
        """
        simulate the outside stimulation of neuron

        :param inject_current: inject current, function of time
        :return: none
        """
        self.u_now += self.inject_current

    def isSpike(self):
        if self.u_now >= self.u_threshold:
            self.spike_status = 2
            self.u_now = self.u_rest
        elif self.u_now > self.u_rest:
            self.spike_status = 1
        elif self.u_now == self.u_rest:
            self.spike_status = 0
        else:
            self.spike_status = -1

    def leaky(self):
        if self.u_now > self.u_rest:
            self.u_now -= self.leaky_rate

    def timeUnit(self):
        self.stimulateNeuron()
        print("n_now = ",self.u_now)
        self.leaky()
        print("n_now = ",self.u_now)
        self.isSpike()
        print("n_now = ",self.u_now)

    def printFigure(self):
        fig, ax = plt.subplots()
        self.u_now = ( 2 * _Rm - 1 ) / ( _Rm * _Cm ) * 0.5 * self.time * self.time
        ax.plot(self.time, self.u_now)
        ax.set(xlabel = 'Time Unit', ylabel = 'Membrane Potential', 
               title = 'LIT Neuron Model')
        ax.grid()
        plt.show()
        
    def simulate(self):
        for i in range(1,5):
            self.timeUnit()
        self.printFigure()



if __name__ == "__main__":
    u_rest = -65
    u_threshold = u_rest + 10
    inject_current = 2
    leaky_rate = 1
    time_try = np.arange(0, 1000, 1)
    i = np.arange(0, 10, 1)
    time_try = np.sin(i)
    neuron = LITModel(u_rest, u_threshold, inject_current, leaky_rate, time_try)
    neuron.simulate()



