from NeuroTools import signals

from numpy import *
from scipy import *
from inference import MLEstimator

def load_experiment(prefix):
  ''' Get spike trains and stimulus, return LikelihoodModel-compatible
      X and Y as tuple (X,Y)'''
  
  # load files
  stimulus = signals.load_spikelist("%s_S.ras" % prefix) 
  excite   = signals.load_spikelist("%s_E.ras" % prefix)
  inhibit  = signals.load_spikelist("%s_I.ras" % prefix)
  t_stop   = max([stimulus.t_stop, excite.t_stop, inhibit.t_stop]) + 1
  
  # convert to correct representation
  stimulus = spike_list_to_matrix(stimulus, t_stop)
  inhibit  = spike_list_to_sparse(inhibit)
  excite   = spike_list_to_sparse(excite)
 
  # combine spike trains
  excite.extend(inhibit)
  return (1.0, t_stop, stimulus, excite)

def analyze_experiment(prefix, model):
  ''' Loads the results of a simulation, performs ML inference on the given model,
  and returns the resulting parameters '''
  
  (duration, timestep, stim, spike) = load_experiment(prefix)
  model.set_data(duration, timestep, stim, spike)
  
  initial   = model.random_args()
  estimator = MLEstimator(model)
  maximized = estimator.maximize(*initial)

  return maximized

def spike_list_to_matrix(list, t_stop=None):
  ''' Converts a spike list into a numpy matrix of width T
      and height = # of trains '''
  N      = len(list.id_list())
  spikes = zeros((N,t_stop))
  for i,v in enumerate(list.spiketrains):
    spikes[i,:] = spike_train_to_matrix(list.spiketrains[v], t_stop)
  return spikes

def spike_train_to_matrix(train, t_stop=None):
  ''' Converts a single spike train to a binary row vector '''
  spikes  = zeros((1, t_stop))
  indices = spike_train_to_indices(train)
  spikes[0,indices] = 1
  return spikes

def spike_train_to_indices(train):
  ''' Casts a train of floating-point spike times into an array
  of integers '''
  return array(train.spike_times, dtype=int32)

def spike_list_to_sparse(list):
  ''' Converts a spike list into a sparse list representation
      with [[s1t1, s1t2,...],[s2t1,s2t2,...],...] '''
  trains = []
  for i,v in enumerate(list.spiketrains):
    trains.append(spike_train_to_indices(list.spiketrains[v]).tolist())
  return trains
