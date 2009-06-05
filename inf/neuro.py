import numpy as np
import scipy.optimize as opt
import random as rd
import math
from memoize import *

def cos_stimulus(time,period):
  return np.asarray(map(lambda t: math.cos(t*math.pi*2/period)**2,range(0,time)))

def spike_train(stim):
  return np.asarray(map(lambda t: rd.random() < t,stim))

def spike_times(train):
  return filter(lambda t: train[t], range(train.size))

def cos_filter(size):
  base = map(lambda i: i * math.pi / (2*size), range(0,size))
  return map(lambda i: math.cos(i), base)

def rand_filter(size):
  return [rd.random() for i in range(size)]

def data_slicer(size, stim, spikes):
  start = lambda t: max(t-size,0)
  sliced =lambda t: np.concatenate(([1], stim[start(t):t], spikes[start(t):t])) 
  return Memoize(lambda t: np.asmatrix(sliced(t)))

def filt_slice(mu, stim_filter, hist_filter):
  sliced = np.concatenate(([mu], stim_filter, hist_filter))
  return np.asmatrix(sliced)

def logI(t, theta, data):
  return theta * data(t).T

def logL(theta, data, delta, time, size, sp_times):
  intensities = [logI(t, theta, data) for t in range(size,time)]
  term1 = sum([intensities[t-size] for t in sp_times])
  term2 = sum([delta*math.exp(intensities[t-size]) for t in range(size,time)])
  return np.array(term1-term2)

def logL_grad(theta, data, delta, time, size, sp_times):
  term1 = sum([data(t) for t in sp_times])
  term2 = sum([delta * math.exp(logI(t,theta,data)) * data(t) for t in range(size,time)])
  result = np.array(term1-term2)
  return np.reshape(result, [result.size])

def logL_hess(theta, data, delta, time, size, sp_times):
  datm = lambda t: np.matrix(data(t))
  dsqu = lambda t: datm(t).T * datm(t)
  result = -1 * delta * sum([dsqu(t)*math.exp(logI(t,theta,data)) for t in range(size,time)])
  return np.asarray(np.reshape(result,[theta.size,theta.size]))

def logL_hess_p(theta, p, *args):
  hessian = np.asmatrix(logL_hess(theta,*args))
  p = np.asmatrix(p)
  return np.asarray(hessian * p)

def maxL(opt, delta, size, spikes, stim):
  theta = filt_slice(1, np.zeros(size), np.zeros(size))
  times = spike_times(spikes)
  data = Memoize(data_slicer(size, stim, spikes))
  args = (data, delta, spikes.size, size, times)
  f = lambda theta: -1 * logL(theta, *args)
  fp = lambda theta: -1 * logL_grad(theta, *args)
  fh = lambda theta, p: -1 * logL_hess_p(theta, p, *args)
  return opt(f=f, x0=theta, fprime=fp, fhess_p=fh)
  
  
def testopt():
  time = 1000
  size = 40
  stim = cos_stimulus(time,200)
  spikes = spike_train(stim)
  times = filter(lambda t: t > size, spike_times(spikes))
  sf = cos_filter(size)
  hf = cos_filter(size)
  delta = 1
  theta = filt_slice(1, sf, hf)
  data = data_slicer(size, stim, spikes)
  args = (data, delta, time, size, times)
  hess_p = lambda th, p, *args: logL_hess(th, *args) * p
  x = opt.fmin_ncg(logL, theta, fprime=logL_grad, fhess_p=hess_p, args=args)
  return x

