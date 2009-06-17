import numpy as np
import math as m
import scipy.ndimage as nd

def rand_stim(time):
  return np.asarray(map(lambda t: rd.random(),range(0,time)))

def rand_train(stim):
  return np.asarray(map(lambda t: rd.random() < t,stim))

def spike_times(train):
  return filter(lambda t: train[t], range(train.size))
