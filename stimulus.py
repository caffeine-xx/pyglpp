from numpy.random import poisson
import random as rd
import math
import numpy as np
from inference import *

class Stimulator:
  """ Stimulator class.  Supports both incremental and 
  range-based stimulus generation.  Returns 
  Nx by T sized stimulus arrays (T=1 for incremental) """

  def __init__(self, delta, Nx):
    self.d = delta
    self.T = 0
    self.Nx = Nx
    self.X = np.zeros([Nx, 100])

  def next(self):
    raise Exception("Not implemented.")
  
  def range(self,t0,t1):
    raise Exception("Not implemented.")


class CosineStim(Stimulator):
  """ It's all about the cosine.  Periods and phases can be specified
  but are usually zero - multiples of 2pi."""

  def __init__(self, delta, Nx, periods=[1], phases=[0]):
    assert len(periods)==len(phases)
    if (len(periods)==1):
      periods = Nx * periods
      phases = Nx * phases
    assert len(periods)==Nx
    Stimulator.__init__(self,delta, Nx)
    self.pe = periods
    self.ph = phases

  def cosine(self,period=(math.pi*2),phase=0):
    return np.vectorize(lambda t: math.cos((phase+t)*math.pi*2*period)**2)

  def simulate(self, time):
    return np.asarray([self.cosine(pe,ph)(time) for pe,ph in zip(self.pe,self.ph)])
  
  def range(self, t0, t1):
    tau = np.arange(t0, t1, self.d)
    return self.simulate(tau)

  def next(self):
    self.T += self.d
    return self.simulate(self.T)

class RandomStim(Stimulator):
  """ Uniform stimulator """

  def range(self, t0, t1):
    return np.random.rand(self.Nx,(t1-t0)/self.d)

  def next(self):
    return np.random.rand(self.Nx,1)

class PoissonSimulator:
  """ Multi-neuron poisson simulator """
  def __init__(self, delta=0.1, stimulus=None, neurons=None, stim_basis=None, spike_basis=None):
    self.delta  = delta
    self.tau    = tau
    self.stim   = stimulus
    self.Nx     = stimulus.shape[0]
    self.N      = neurons
    self.spikes = np.zeros([self.N,tau.size+1])
    self.T      = tau.size
    self.I      = np.zeros([self.N,tau.size+1])
    self.st_bas = stim_basis
    self.sp_bas = spike_basis
    self.b_spikes= np.zeros([self.N,  stimulus.shape[1], self.sp_bas.shape[0]])
    self.b_stims = run_bases(stim_basis, stimulus)
    self._rebase(0, tau.size+1)

  def _rebase(self, start, end):
    """ Private method: recalculates basis versions of spike  
        between start and end times, using prev. history if
        available  """
    # TODO: Wildly inefficient, needs to be incremental.
    t0 = max(0,start-self.tau.size)
    t1 = end
    spikes = self.spikes[:,t0:t1]
    bspikes = run_bases(self.sp_bas, spikes)
    self.b_spikes[:,t0:t1,:] = bspikes

  def _grow(self,t):
    """ Adds zeros as necessary """
    while (self.I.shape[1] < t):
      self.I = np.hstack((self.I, np.zeros((self.I.shape[0], self.T))))
    while (self.spikes.shape[1] < t):
      self.spikes = np.hstack((self.spikes, np.zeros((self.spikes.shape[0], self.T))))
    while (self.stim.shape[1] < t):
      self.stim = np.hstack((self.stim, np.zeros((self.stim.shape[0], self.T))))

  def _forward(self,t):
    """ Private method for simulating forward in time the intensities """
    self._grow(t)
    while self.T < t:
      for i in range(0,self.N):
        for j in range(0,self.Nx):
          self.I[i,self.T] += np.sum(self.K[i,j,:] * self.b_stims[j,self.T,:])
        for j in range(0,self.N):
          self.I[i,self.T] += np.sum(self.H[i,j,:] * self.b_spikes[j,self.T,:])
        self.I[i,self.T] += self.Mu[i]
        self.I[i,self.T] = np.ma.exp(self.I[i,self.T])
        self.spikes[i,self.T] = self._poisson_spike(self.I[i,self.T])
      self._rebase(self.T,self.T+1)
      self.T += 1
  
  def _poisson_spike(self,I):
    """ Just spike 'er. """
    return np.random.poisson(I,1)>0

  def simulate(self, K, H, Mu, t):
    """ Run simulation until time T and return the whole spike train"""
    self.K, self.H, self.Mu = K, H, Mu
    self._forward(t)
    return self.spikes

  def sparse_spikes(self):
    return [spike_times(self.spikes[i,:]) for i in xrange(self.N)]

def rand_stim(time):
  return np.asarray(map(lambda t: rd.random(),range(0,time)))

def rand_train(stim):
  return np.asarray(map(lambda t: rd.random() < t,stim))

def spike_times(train):
  return filter(lambda t: train[t], range(train.size))
