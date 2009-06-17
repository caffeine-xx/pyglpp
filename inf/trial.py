from numpy import *
from pyNN.nest import *

from trains import *
from likelihood_model import MultiNeuron
'''
Params:
  - timestep 
  - duration 
  - spike_file
  - stim_file
Added on by trial:
  - spikes  = spike trains of neurons (sparse)
  - stims   = timeseries of stimuli (full)
'''
def exp_two_neuron(params):
  stim_file = 'results/exp_2n_stim.dat'
  spike_file = 'results/exp_2n_spike.dat'
  
  simparams = dict()
  simparams['timestep'] = params['timestep']
  simparams['duration'] = params['duration']
  simparams['min_delay'] = 1.0
  simparams['max_delay'] = 4.0

  setup(**simparams)
 
  cells = Population(2,IF_cond_alpha)

  stimulus = gen_stimulus(params['stimuli'], params['duration'])
  stim_spikes = mspike_times(stimulus)
  
  source = create(SpikeSourceArray, cellparams={'spike_times':stim_spikes[0]})
  
  connect(source, cells, weight=1.0, delay=1.0, synapse_type="excitatory")
  
  I_Connector = FixedProbabilityConnector(1.0, weights=-.2, delays=1.0)
  I_to_E = Projection(cells, cells, I_Connector, target="inhibitory") 

  record(source, stim_file)
  record(cells, spike_file)
  
  run(params['duration'])

  end()
  return stim_file, spike_file

def ml_trial(model, params):
  from NeuroTools import io
  model.set_params(params)

def gen_stimulus(stimuli, duration):
  stim = random.random((stimuli, duration))
  stim = (stim < random.random((stimuli,duration)))
  return stim


if __name__=="__main__":

  params = {
    'duration':1000.0,
    'timestep':1.0,
    'stimuli':2
  }

  print exp_twoneuron(params)
