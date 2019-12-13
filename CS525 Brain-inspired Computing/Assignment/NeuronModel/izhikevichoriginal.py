# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:47:11 2016

@author: arthur

Izhikevich neuron - Original.
"""

from pylab import *

#1) Initialize parameters.
tmax = 1000
dt = 0.5

#1.1) Neuron/Network pairs.
a = 0.02
b = 0.2
c = -65
d = 8

#1.2) Input pairs
lapp = 10
tr = array([200, 700])/dt  #stm time
 
#2) Reserve memory
T = ceil(tmax / dt)
v = zeros(T)
u = zeros(T)
v[0] = -70 #Resting potential
u[0] = -14 #Steady state
 
#3) For-loop over time.
for t in arange(T-1):
#3.1) Get input.
    if t > tr[0] and t < tr[1]:
        l = lapp
    else:
        l = 0
    if v[t] < 35:
        #3.2) Update DOE.
        dv = (0.04*v[t]+5)*v[t]+140-u[t]
        v[t+1] = v[t]+(dv+l)*dt
        du = a*(b*v[t]-u[t])
        u[t+1] = u[t] + dt*du
    else:
        #3.3) Spike!
        v[t] = 35
        v[t+1] = c
        u[t+1] = u[t] + d
        
#4) Plot voltage trace
figure()
tvec = arange(0, tmax, dt)
plot(tvec, v, 'b', label='Voltage trace')
xlabel('Time[ms]')
ylabel('Membrane voltage [mV]')
title('A single qIF neuron with current step input')
show()
        
print(FIN)