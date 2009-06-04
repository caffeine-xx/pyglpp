import numpy as np
import random as rd
import math

def cos_stimulus(time,period):
  return map(lambda t: math.cos(t*math.pi*2/period)**2,range(0,time))

def spike_train(stim):
  return map(lambda t: rd.random() < t,stim)

def spike_times(train):
  times = []
  for t, s in enumerate(train):
    if s:
      times.append(t)
  return times

def filter(t, filt, data):
  result = 0
  for i, f in enumerate(filt):
    result = result + f * data[t - i]
  return result

def log_intensity(t, data, filters):
  result = 0
  for i, stim in enumerate(data):
    filt = filters[i]
    result = result + filter(t, filt, stim)
  return result

def log_likelihood(spikes, stimulus, hist_filt, stim_filt):
  data = [spikes, stimulus]
  filt = [hist_filt, stim_filt]
  start = min(len(hist_filt), len(stim_filt)) + 1
  times = filter(lambda t: t > start, spike_times(spikes))
  int = map(lambda time: log_intensity(time, data, filt), times)
  return sum(int)

def cosfilter(size):
  base = map(lambda i: i * math.pi / (2*size), range(0,size))
  return map(lambda i: math.cos(i), base)

