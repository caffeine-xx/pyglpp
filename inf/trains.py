from numpy.random import poisson
# from slicers import *
import random as rd
import math
import numpy as np
def cos_stim(time,period):
  return np.asarray(map(lambda t: math.cos(t*math.pi*2/period)**2,range(0,time)))

def rand_stim(time):
  return np.asarray(map(lambda t: rd.random(),range(0,time)))

"""def poiss_train(theta,stim,delta,size):
  time = stim.size
  spikes = np.zeros(time)
  times = []
  data = data_slicer(size,stim,spikes)
  inten = lambda t: np.dot(theta, data(t).T)
  spikes = np.asarray([(poisson(delta*inten(t-1),1)>0)[0] for t in range(size+1,time)])
  times = spike_times(spikes)
  return spikes, times
"""

def rand_train(stim):
  return np.asarray(map(lambda t: rd.random() < t,stim))

def spike_times(train):
  return filter(lambda t: train[t], range(train.size))
