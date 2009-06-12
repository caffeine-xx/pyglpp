from numpy.random import poisson
import random as rd
import math
import numpy as np
import basis as b

class PoissonSimulator:
  """ Multi-neuron poisson simulator """
  def __init__(self, delta, tau, stimulus, neurons, stim_basis, spike_basis):
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
    self.b_stims = b.run_bases(stim_basis, stimulus)
    self._rebase(0, tau.size+1)

  def _rebase(self, start, end):
    """ Private method: recalculates basis versions of spike  
        between start and end times, using prev. history if
        available  """
    # TODO: Wildly inefficient, needs to be incremental.
    t0 = max(0,start-self.tau.size)
    t1 = end
    spikes = self.spikes[:,t0:t1]
    bspikes = b.run_bases(self.sp_bas, spikes)
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


def rand_stim(time):
  return np.asarray(map(lambda t: rd.random(),range(0,time)))

def rand_train(stim):
  return np.asarray(map(lambda t: rd.random() < t,stim))

def spike_times(train):
  return filter(lambda t: train[t], range(train.size))
