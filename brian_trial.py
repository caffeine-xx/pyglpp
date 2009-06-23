from brian import *
from brian.library.IF import *
from brian.library.synapses import *
import time

def single_izhikevich_trial():
  # Izhikevich model with excitatory synapse
  model = Izhikevich()
  reset = AdaptiveReset(Vr=-75*mV, b=0.2/ms)
  neuron = NeuronGroup(10,model=model, threshold=-30.0*mV, reset=reset)
  # Poisson stimulus
  rates = lambda t: max(0,normal(40.0*Hz, 20.0*Hz))
  stimulus = PoissonGroup(1,rates=rates)
  connection = Connection(stimulus, neuron)
  connection[0,:] = 80*mvolt
  # Spike recording
  neuron_monitor = SpikeMonitor(neuron,record=True)
  input_monitor = SpikeMonitor(stimulus,record=True)
  run(1000*ms)
  raster_plot(input_monitor)
  show()
  return neuron
