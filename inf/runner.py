
import numpy as np
from neuro import *
from trains import *
from basis import *
import likelihood_model as lm 
import mlestimator as ml

reload(lm)
reload(ml)


def test1():
  
  # basic parameters
  neurons = 10
  stimuli = 2
  delta = 0.1
  time = np.arange(0, 10, delta)
  tau = np.arange(0, 2, 0.1)
  
  # simulation
  stim = np.zeros([stimuli,time.size])+0.5
  spikes = np.zeros([neurons,time.size])
  sparse = neurons*[[]]
  for i in range(0,neurons):
    spikes[i,:] = rand_train(stim[0,:])
    sparse[i] = spike_times(spikes[i,:])
  
  # initialization
  basis = straight_basis(0.5)
  K = np.random.random([neurons,stimuli,4])
  H = np.random.random([neurons,neurons,4])
  Mu = np.random.random([neurons])

  # run it
  mn = lm.MultiNeuron(delta, tau, stim, spikes, sparse, basis)
  mle = ml.MLEstimator(mn)
  K, H, Mu = mle.maximize(K, H, Mu)

  return K, H, Mu
  
