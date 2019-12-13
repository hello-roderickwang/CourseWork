#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import math


class Unit:
    def __init__(self):
        self.ms = 1
        self.s = 10
        self.mv = 1
        self.v = 10
        self.m_omega = 1
        self.pf = 1

class LIF_Neuron:
    """
    u(t) = u_rest + membrane_resistance * stimulate_current * ( 1 - e ^ ( -t / membrane_resistance * membrane_capacity ))
    """
    def __init__(self, stimulate_current, stimulate_time):
        unit = Unit()
        '''
        self.rest_potential = -70 * unit.mv
        self.membrane_potential = self.rest_potential
        self.membrane_resistance = 200 * unit.m_omega
        self.membrane_capacity = 80 * unit.pf
        self.spike_threshold = -50 * unit.mv
        self.spike_time = 5 * unit.ms
        self.stimulate_period = 8 * unit.ms
        '''
        self.rest_potential = -70 * unit.mv
        self.membrane_potential = self.rest_potential
        self.membrane_resistance = 10 * unit.m_omega
        self.membrane_capacity = 4 * unit.pf
        self.spike_threshold = -50 * unit.mv
        self.spike_time = 5 * unit.ms
        self.stimulate_period = 0.4 * unit.ms
        
        self.neuron_status = 0
        self.stimulate_current = stimulate_current
        self.stimulate_time = stimulate_time
        self.history = np.array([])
        self.counter = 0
    
    def is_spike(self):
        if self.membrane_potential >= self.spike_threshold:
            self.neuron_status = 2
            self.membrane_potential = self.rest_potential
            print("membrane_potential=",self.membrane_potential)
            print("-----Neuron Spikes-----")
        elif self.membrane_potential > self.rest_potential:
            self.neuron_status = 1
            print("membrane_potential=",self.membrane_potential)
        elif self.membrane_potential == self.rest_potential:
            self.neuron_status = 0
            print("membrane_potential=",self.membrane_potential)
        else:
            self.neuron_status = -1
            print("membrane_potential=",self.membrane_potential)
    
    #def stimulate_neuron(self, stimulation, time):
    def stimulate_neuron(self, stimulation, time):
        if stimulation.size != 0:
            #time = 3
            return np.array([self.membrane_potential + self.membrane_resistance * stimulation * ( 1 - math.e ** ( -1 * stimulation ) / ( self.membrane_resistance * self.membrane_capacity))])
        else:
            print("Stimulation is empty!!!")
        
    """
    def judge_status(self):
        if self.neuron_status == 2:
            self.membrane_potential = self.rest_potential
            print("Neuron Spike!")
        elif self.neuron_status == 1:
            print("Not enough for spike!")
        elif self.neuron_status == 0:
            print("Not enough to change the rest potential!")
        else:
            print("Neuron has been depressed!")
    """
        
    def neuron_period(self):
        last_membrane_potential = 0
        #last_membrane_potential_difference = 0
        for i in range(0, self.stimulate_time):
            '''
            if self.neuron_status == 1:
                #self.history = np.append(self.history, self.membrane_potential + last_membrane_potential_difference, axis = None)
                self.history = np.append(self.history, self.membrane_potential, axis = None)
                #last_membrane_potential_difference = 0
            else:
                self.history = np.append(self.history, self.membrane_potential, axis = None)            
            '''
            print("membrane_potential:", self.membrane_potential)
            print("self.history:", self.history)
            stimulation = self.stimulate_current[i]
            print("stimulation:", stimulation)
            time = i
            self.membrane_potential = self.stimulate_neuron(stimulation, time)
            """2019-2-22 2:38 RW
            upper line is the key, i need to extract quality of each augment
            i still think there are some questions about the time scale
            i should try to segment time into small unit and try to control every movement based
            on these segmented time pieces
            """
            
            if last_membrane_potential != self.membrane_potential:
                self.history = np.append(self.history, self.membrane_potential, axis = None)
            else:
                self.counter += 1
            
            self.is_spike()
            
            self.history = np.append(self.history, self.membrane_potential, axis = None)
            
            #last_membrane_potential_difference = self.membrane_potential - last_membrane_potential
            last_membrane_potential = self.membrane_potential
            
            
            #self.judge_status()
            #self.history = np.append(self.history, self.membrane_potential, axis = None)

        
if __name__ == "__main__":
    time = 20
    current = np.full(10, 0.7, dtype = float)
    current = np.insert(current, 0, np.full(5, 0, dtype = float), axis = None)
    current = np.append(current, np.full(5, 5, dtype = float), axis = None)
    print("current:", current)
    lif_neuron = LIF_Neuron(current, time)
    lif_neuron.neuron_period()
    plt.plot(range(0, 2 * time - lif_neuron.counter), lif_neuron.history)
    plt.title("LIF Neuron Model")
    plt.xlabel("Time Scale")
    plt.xscale('linear')
    plt.ylabel("Membrane Potential")
    plt.yscale('linear')
    plt.axhline(y = lif_neuron.rest_potential, color = 'g')
    plt.axhline(y = lif_neuron.spike_threshold, color = 'r')
    plt.show()
    
"""2019-2-21 2:33 RW
membrane_potential and stimulate_potential should be the same and they all should be an int value

need to add for loop to make time scale

adjust the value of the coefficient
"""

"""2019-2-21 16:09 RW
there should be seperately two parts of a spike
the formula i have is a charge? rest euality
the is a reset one
and time line should be re-done
one neuron cycle is about 10ms
and one spike only take 1ms
"""
    





