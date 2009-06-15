import numpy as np
import nest as ns



class NestSimulator:
  """ Only a single simulator can run at a time (per process?) because
  the kernel gets reset """

  def __init__(self, delta, type, conn):
    nest.ResetKernel()

    # Initialize the parameters of the integrate and fire neuron
    tauSyn = 0.5
    tauMem = 20.0
    theta  = 20.0
    J      = 0.1 # postsynaptic amplitude in mV
    neuron_params= {"C_m"       : 1.0,
                    "tau_m"     : tauMem,
                    "tau_syn_ex": tauSyn,
                    "tau_syn_in": tauSyn,
                    "t_ref"     : 2.0,
                    "E_L"       : 0.0,
                    "V_th"      : theta}
    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    
    # Build the network
    nodes  = nest.Create("iaf_psc_alpha",conn.N) 
    
    
    spikes = nest.Create("spike_detector")
    
    nest.ConvergentConnect(range(1,conn.N),spikes,model="excitatory")
    
    for i in range(0,conn.N):
      nest.ConvergentConnect(conn.conn[i]+1,[i+1],model="excitatory")

    self.sim = nest
  
  def run(self,time):
    self.sim.Simulate(time)

