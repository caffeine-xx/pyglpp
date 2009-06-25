from NeuroTools import signals

from numpy import *
from scipy import *
from scipy import io
from inference import MLEstimator

def analyze_experiment(model,experiment):
  ''' Loads the results of a simulation, performs ML inference on the given model,
  and returns the resulting parameters '''
  
  (timestep, duration, stim, spike) = experiment
  model.set_data(timestep, duration, stim, spike)
  
  initial   = model.random_args()
  estimator = MLEstimator(model)
  maximized = estimator.maximize(*initial)
  
  return maximized

def load_brian_experiment(prefix):
  ''' Get spike trains from a Brian experiment, which only has two files 
      Converts all times to whole integers.  Total time becomes max_time
      divided by dt.'''

  # load preliminaries
  file = signals.StandardTextFile("%s_S.ras" % prefix)
  file._StandardTextFile__read_metadata()
  dt   = file.metadata['dt']
 
  # load files
  stim = signals.load_spikelist("%s_S.ras" % prefix)
  neur = signals.load_spikelist("%s_N.ras" % prefix)
  t_stop = int(max(stim.t_stop, neur.t_stop)/dt) + 10

  # convert
  stim = spike_list_to_matrix(stim, t_stop, dt)
  neur = spike_list_to_sparse(neur, dt)
  
  # done
  return (1.0, t_stop, stim, neur)

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

def param_to_dict(parameters):
  return dict(zip(('K','H','Mu'),parameters))

def param_from_dict(p):
  return (p['K'],p['H'],p['Mu'])

def save_parameters(filename, parameters):
  ''' Saves a set of poisson parameters to a file '''
  io.savemat(filename,param_to_dict(parameters))

def load_parameters(filename):
  ''' Loads a set of poisson parmaeters from a file '''
  p = io.loadmat(filename)
  return param_from_dict(p)

def spike_list_to_matrix(list, t_stop=None, dt=1.0):
  ''' Converts a spike list into a numpy matrix of width T
      and height = # of trains '''
  N      = len(list.id_list())
  spikes = zeros((N,t_stop))
  for i,v in enumerate(list.spiketrains):
    spikes[i,:] = spike_train_to_matrix(list.spiketrains[v], t_stop,dt)
  return spikes

def spike_train_to_matrix(train, t_stop=None, dt=1.0):
  ''' Converts a single spike train to a binary row vector '''
  spikes  = zeros((1, t_stop))
  indices = spike_train_to_indices(train,dt)
  spikes[0,indices] = 1
  return spikes

def spike_train_to_indices(train,dt=1.0):
  ''' Casts a train of floating-point spike times into an array
  of integers '''
  return array(array(train.spike_times)/dt, dtype=int32)

def spike_list_to_sparse(list,dt=1.0):
  ''' Converts a spike list into a sparse list representation
      with [[s1t1, s1t2,...],[s2t1,s2t2,...],...] '''
  trains = []
  for i,v in enumerate(list.spiketrains):
    trains.append(spike_train_to_indices(list.spiketrains[v],dt=1.0).tolist())
  return trains
