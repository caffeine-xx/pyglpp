#! /usr/bin/env python

'''
This script simulates a neuron driven by an excitatory and an inhibitory
population of neurons firing Poisson spike trains. The aim is to find a
firing rate for the inhibitory population that will make the neuron fire
at the same rate as the excitatory population.

Optimization is performed using the bisection method from Scipy, simulating 
the network repeatedly.
'''

import nest
import nest.voltage_trace
from scipy.optimize import bisect

# suppress info messages
nest.sr("M_WARNING setverbosity")
nest.ResetKernel()

t_sim = 25000.0    # how long we simulate
n_ex  = 16000      # size of the excitatory population
n_in  =  4000      # size of the inhibitory population
r_ex  =     5.0    # mean rate of the excitatory population
r_in  =    20.5    # initial rate of the inhibitory population
epsc  =    45.0    # peak amplitude of excitatory synaptic currents
ipsc  =   -45.0    # peak amplitude of inhibitory synaptic currents
d     =     1.0    # synaptic delay
lower =    15.0    # lower bound of the search interval
upper =    25.0    # upper bound of the search interval
prec  =     0.01   # how close need the excitatory rates be

neuron        = nest.Create("iaf_neuron")
noise         = nest.Create("poisson_generator",2)
voltmeter     = nest.Create("voltmeter")
spikedetector = nest.Create("spike_detector")

nest.SetStatus(noise, [{"rate": n_ex*r_ex}, {"rate": n_in*r_in}])

# Record potential every 10ms. We must record gid and time for later plotting
# with voltage_trace.
nest.SetStatus(voltmeter, {"interval": 10.0, "withgid": True, "withtime": True})

nest.ConvergentConnect(noise, neuron, [epsc, ipsc], 1.0)
nest.Connect(voltmeter, neuron)
nest.Connect(neuron, spikedetector)

def output_rate(guess):
    print "Inhibitory rate estimate: %5.2f Hz ->" % guess,   
    rate = float(abs(n_in*guess))
    nest.SetStatus([noise[1]], "rate", rate)
    nest.SetStatus(spikedetector, "n_events", 0)
    nest.Simulate(t_sim)
    out=nest.GetStatus(spikedetector, "n_events")[0]*1000.0/t_sim
    print "Neuron rate: %6.2f Hz (goal: %4.2f Hz)" % (out, r_ex)
    return out

# Find a rate for the inhibitory population such that the neuron
# fires at the same rate as the excitatory population
in_rate = bisect(lambda x: output_rate(x)-r_ex, lower, upper, xtol=prec)

print "\nOptimal rate for the inhibitory population: %.2f Hz" % in_rate

nest.voltage_trace.from_device(voltmeter)
