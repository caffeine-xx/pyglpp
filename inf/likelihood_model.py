import numpy as np
import scipy.optimize as opt
import basis as b
import math as m
from memoize import Memoize

class LikelihoodModel:

  def logL(self,*args):
    raise Exception('not implemented')

  def logL_grad(self,*args):
    raise Exception('not implemented')

  def logL_hess_p(self,*args):
    raise Exception('not implemented')

  def unpack(self,theta, args):
    raise Exception('not implemented')

  def pack(self,*a):
    raise Exception('not implemented')

class MultiNeuron(LikelihoodModel):
  """ Multi-neuron Poisson model with user-specified bases"""

  def __init__(self, delta, tau, stimulus, spikes, sparse, stim_basis, spike_basis):
    self.N, self.T = spikes.shape
    self.Nx = stimulus.shape[0]
    self.ind = range(tau.size+1,self.T)
    self.spikes = spikes
    self.stims  = stimulus
    self.delta  = delta
    self.tau    = tau 
    self.sparse = [filter(lambda t: t > tau.size, sparse[i]) for i in range(0,self.N)]
    self.spike_b= spike_basis.shape[0]
    self.stim_b = stim_basis.shape[0]
    self.base_spikes = b.run_bases(spike_basis, spikes)
    self.base_stims  = b.run_bases(stim_basis, stimulus)

  def pack(self, K, H, Mu):
    shapes = (K.size, K.shape, H.size, H.shape, Mu.size, Mu.shape)
    theta = np.r_[np.ndarray.ravel(K),np.ndarray.ravel(H),np.ndarray.ravel(Mu)]
    return theta, shapes

  def unpack(self, theta, *args):
    (Ksize, Kshape, Hsize, Hshape, Musize, Mushape) = args[0]
    K = np.ndarray.reshape(theta[0:Ksize], Kshape)
    H = np.ndarray.reshape(theta[Ksize:Ksize+Hsize], Hshape)
    Mu= np.ndarray.reshape(theta[(Ksize+Hsize):Ksize+Hsize+Musize], Mushape)
    return K, H, Mu

  def logI(self, K, H, Mu):
    I = np.zeros([self.N,self.T])
    for i in range(0,self.N):
      for j in range(0,self.Nx):
        I[i,:] += sum([self.base_stims[j,:,l] * K[i,j,l] for l in range(0,self.stim_b)])
      for j in range(0,self.N):
        I[i,:] += sum([self.base_spikes[j,:,l] * H[i,j,l] for l in range(0,self.spike_b)])
      I[i,:] += Mu[i]
    return I

  def logL(self, K, H, Mu):
    I = self.logI(K,H,Mu)
    t1 = 0
    for i in range(0,self.N):
      t1 += sum([I[i,t] for t in self.sparse[i]])
    t2 = np.sum(np.sum(np.ma.exp(I)))
    return t1 - self.delta*t2

  def logL_grad(self, K, H, Mu):
    I = self.logI(K,H,Mu)
    dK = np.zeros([self.N, self.Nx, self.stim_b])
    dH = np.zeros([self.N, self.N, self.spike_b])
    dM = np.zeros([self.N])
    for i in range(0,self.N):
      for j in range(0,self.Nx):
        dK[i,j,:] = sum([self.base_stims[j,t,:] for t in self.sparse[i]])
        dK[i,j,:] -= self.delta * np.sum(self.base_stims[j,:,:] * np.ma.exp(I[i,:]).reshape((I[i,:].size,1)),0)
      for j in range(0,self.N):
        dH[i,j,:] = sum([self.base_spikes[j,t,:] for t in self.sparse[i]])
        dH[i,j,:] -= self.delta * np.sum(self.base_spikes[j,:,:] * np.ma.exp(I[i,:]).reshape((I[i,:].size,1)),0)
      t1 = len(self.sparse[i])
      dM[i]= len(self.sparse[i])-self.delta*np.sum(I[i,:])
    return dK, dH, dM

