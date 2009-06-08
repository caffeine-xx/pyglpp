import numpy as np

def data_slicer(size, stim, spikes):
  start = lambda t: max(t-size,0)
  sliced =lambda t: np.concatenate(([1], stim[start(t):t], spikes[start(t):t])) 
  return (lambda t: np.asmatrix(sliced(t)))

def filt_slice(mu, stim_filter, hist_filter):
  sliced = np.concatenate(([mu], stim_filter, hist_filter))
  return np.asmatrix(sliced)
