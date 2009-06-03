#! /usr/bin/env python

# This version uses NEST's RandomConvergentConnect functions.
# It used integrate and fire neurons with delta shaped postsynaptic currents
# (exponential postsynaptic potentials) as in Brunel 2000.


import nest, nest.raster_plot
import time

nest.ResetKernel()

startbuild = time.time()

nest.SetKernelStatus({"resolution": 0.1,
                      "print_time": True})

# Define simulation duration, network size and number of recorded neurons
simtime = 100.0 # Simulation time in ms
order = 2500
NE = 4*order
NI = 1*order
N_neurons = NE+NI
N_rec = 50 # record from 50 neurons

# Initialize the parameters of the integrate and fire neuron
tauMem = 20.0
theta = 20.0

# Define connection parameters
J_ex = 0.1
g = 5.0
J_in = -g*J_ex
epsilon = 0.1    # connection probability
CE = int(epsilon*NE)   # number of excitatory synapses per neuron
CI = int(epsilon*NI)   # number of inhibitory synapses per neuron  
delay = 1.5     # the delay in ms


# Parameters for asynchronous irregular firing
eta = 1.0
nu_th = theta/(J_ex*tauMem)
nu_ex = eta*nu_th
p_rate = 1000.0*nu_ex


print "Creating network nodes..."

neuron_params= {"C_m":    tauMem,
                "tau_m":  tauMem,
                "t_ref":  2.0,
                "E_L":    0.0,
                "V_th":   theta}

nest.SetDefaults("iaf_psc_delta", neuron_params)

nodes_ex=nest.Create("iaf_psc_delta", NE)
nodes_in=nest.Create("iaf_psc_delta", NI)
nodes = nodes_ex + nodes_in

noise=nest.Create("poisson_generator", 1, {"rate": p_rate})

nest.SetDefaults("spike_detector", {"withtime": True,
                                    "withgid": True})

espikes=nest.Create("spike_detector", 1, {"label": "brunel-py-espikes"})
ispikes=nest.Create("spike_detector", 1, {"label": "brunel-py-espikes"})

print "Connecting devices."

nest.SetDefaults("static_synapse", {"delay": delay})
nest.CopyModel("static_synapse", "excitatory", {"weight":J_ex})
nest.CopyModel("static_synapse", "inhibitory", {"weight":J_in})
 
nest.DivergentConnect(noise, nodes, model="excitatory")
nest.ConvergentConnect(nodes_ex[:N_rec], espikes, model="excitatory")
nest.ConvergentConnect(nodes_in[:N_rec], ispikes, model="excitatory")

print "Connecting network."

print "Excitatory connections"
nest.RandomConvergentConnect(nodes_ex, nodes, CE,model="excitatory")

print "Inhibitory connections"
nest.RandomConvergentConnect(nodes_in, nodes, CI,model="inhibitory")

endbuild=time.time()

print "Simulating."

nest.Simulate(simtime)

endsimulate= time.time()

events_ex= nest.GetStatus(espikes,"n_events")[0]
rate_ex= events_ex/simtime*1000.0/N_rec
events_in= nest.GetStatus(ispikes,"n_events")[0]
rate_in= events_in/simtime*1000.0/N_rec

synapses_ex = nest.GetStatus("excitatory", "num_connections")
synapses_in = nest.GetStatus("inhibitory", "num_connections")

build_time= endbuild-startbuild
sim_time  = endsimulate-endbuild

print "Brunel network simulation (Python)"
print "Number of neurons :", N_neurons
print "Number of synapses:", synapses_ex + synapses_in
print "       Exitatory  :", synapses_ex
print "       Inhibitory :", synapses_in
print "Excitatory rate   : %.2f Hz" % rate_ex
print "Inhibitory rate   : %.2f Hz" % rate_in
print "Building time     : %.2f s" % build_time
print "Simulation time   : %.2f s" % sim_time

nest.raster_plot.from_device(espikes, hist=True)
