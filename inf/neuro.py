import numpy as np
import random as rd
import math

# cos^2 stimulus
def cos_stimulus(time,period):
  return np.asarray(map(lambda t: math.cos(t*math.pi*2/period)**2,range(0,time)))

def spike_train(stim):
  return np.asarray(map(lambda t: rd.random() < t,stim))

def spike_times(train):
  times = []
  for t, s in enumerate(train):
    if s:
      times.append(t)
  return times

def cos_filter(size):
  base = map(lambda i: i * math.pi / (2*size), range(0,size))
  return map(lambda i: math.cos(i), base)

def data_slice(t, size, stim, spikes):
  start = max(t-size,0)
  return np.concatenate(([1], stim[start:t], spikes[start:t])) 

def filt_slice(mu, stim_filter, hist_filter):
  return np.concatenate(([mu], stim_filter, hist_filter))

def loglike(delta, mu, stim, stim_filter, spikes, spike_filter):
  theta = filt_slice(mu, stim_filter, spike_filter)
  size = stim_filter.size
  times = spike_times(spikes)
  data = lambda t: data_slice(t, size, stim, spikes)
  logint = lambda t: np.dot(theta,data(t))
  intensities = map(logint, range(spikes.size))
  term1 = sum([intensities[t] for t in times])
  term2 = sum([delta*math.exp(intensities[t]) for t in range(spikes.size)])
  return term1 - term2

def logL(theta, data, delta, time, sp_times):
  intensities = map(lambda t: np.dot(theta,data(t)), range(time))
  term1 = sum([intensities[t] for t in sp_times])
  term2 = sum([delta*math.exp(intensities[t]) for t in range(time)])
  return term1-term2

def logL_grad(theta, data, delta, time, sp_times):
  term1 = sum([data(t) for t in sp_times])
  term2 = sum([delta * math.exp(np.dot(theta, data(t))) * data(t) for t in range(time)])
  return term1 - term2

def logL_hess(theta, data, delta, time, sp_times)
  datm = lambda t: np.matrix(data(t))
  dsqu = lambda t: datm(t) * datm(t).T
  int = lambda t: math.exp( np.dot(theta, data(t)) )
  return -1 * delta * sum([dsqu(t)*int(t) for t in range(time)])

def max_likelihood(delta, size, spikes, stim):
  theta = filt_slice(1, np.zeros(size), np.zeros(size))
  times = spike_times(spikes)
  data = lambda t: data_slice(t, size, stim, spikes)
  

def test():
  stim = cos_stimulus(1000,200)
  spikes = spike_train(stim)
  sf = cos_filter(30)
  hf = cos_filter(50)
  return log_likelihood(spikes, stim, hf, sf)

