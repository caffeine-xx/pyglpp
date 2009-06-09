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
  """ Multi-neuron Poisson model with cosine bases"""

  def __init__(self, delta, tau, stimulus, spikes, sparse, basis):
    # simulation size   
    self.N, self.T = spikes.shape
    self.Nx = stimulus.shape[0]
    self.ind = range(tau.size+1,self.T)
    
    # parameters
    self.spikes = spikes
    self.stims  = stimulus
    self.basis  = basis
    self.delta  = delta
    self.tau    = tau # time vector of a filter 
    
    # get rid of spikes before filter duration
    self.sparse = [filter(lambda t: t > tau.size, sparse[i]) for i in range(0,self.N)]
    
    # slices of spike train or stimulus one time-window wide
    self.spike_slice = lambda t: self.spikes[:,max(0,t-tau.size):t]
    self.stim_slice  = lambda t: self.stims[:,max(0,t-tau.size):t]
    
    # spike-triggered total slices 
    self.st_spike= self.N * [np.zeros([self.N,tau.size])]
    self.st_stim = self.N * [np.zeros([self.Nx,tau.size])]
    for i in range(0,self.N):
      self.st_spike[i] = sum([self.spike_slice(t) for t in self.sparse[i]])
      self.st_stim[i]  = sum([self.stim_slice(t) for t in self.sparse[i]])
    
    # overall total slices
    self.tt_spike= sum([self.spike_slice(t) for t in self.ind])
    self.tt_stim = sum([self.stim_slice(t) for t in self.ind])

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

  def get_filtered(self, i, K, H, Mu):
    k_i = b.filter_builder(K[i,:,:], self.tau, self.basis)
    h_i = b.filter_builder(H[i,:,:], self.tau, self.basis)
    m_i = Mu[i]
    Fk_i = np.sum(b.run_filter(self.stims, k_i),0)
    Fh_i = np.sum(b.run_filter(self.spikes, h_i),0)
    return Fk_i + Fh_i + m_i

  def logL(self, K, H, Mu):
    lam = Memoize(lambda i: self.get_filtered(i, K, H, Mu))
    t1 = sum([b.selective_sum(lam(i),self.sparse[i]) for i in range(0,self.N)])
    t2 = sum([np.sum(np.ma.exp(lam(i))) for i in range(0,self.N)])
    return t1 - self.delta*t2

  def logL_grad(self, K, H, Mu):
    lam = Memoize(lambda i: self.get_filtered(i, K, H, Mu))
    
    dK = np.zeros([self.N, self.Nx, self.tau.size])
    dH = np.zeros([self.N, self.N, self.tau.size])
    dM = np.zeros([self.N])
    
    for i in range(0,self.N):
      for j in range(0,self.Nx):
        t1 = self.st_stim[i][j,:]
        t2 = sum([self.stim_slice(t)[j] * lam(i)[t] for t in self.ind])
        dK[i,j,:] = t1-self.delta*t2
      for j in range(0,self.N):
        t1 = self.st_spike[i][j,:]
        t2 = sum([self.spike_slice(t)[j] * lam(i)[t] for t in self.ind])
        dH[i,j,:] = t1-self.delta*t2
      t1 = len(self.sparse[i])
      t2 = lam(i).sum()
      dM[i]= t1-self.delta*t2
    
    dK = np.sum(dK,0)
    dH = np.sum(dH,0)
    
    return dK, dH, dM

"""
class SingleNeuron(LikelihoodModel):
  def logI(self,t, theta, data):
    return np.reshape(theta * data(t).T,[1])

  def logL(self,theta, data, delta, time, size, sp_times):
    intensities = [logI(t, theta, data) for t in range(size,time)]
    term1 = sum([intensities[t-size] for t in sp_times])
    term2 = sum([delta*math.exp(intensities[t-size]) for t in range(size,time)])
    return np.array(term1-term2)

  def logL_grad(self,theta, data, delta, time, size, sp_times):
    term1 = sum([data(t) for t in sp_times])
    term2 = sum([delta * math.exp(logI(t,theta,data)) * data(t) for t in range(size,time)])
    result = np.array(term1-term2)
    return np.reshape(result, [result.size])

  def logL_hess(self,theta, data, delta, time, size, sp_times):
    datm = lambda t: np.matrix(data(t))
    dsqu = lambda t: datm(t).T * datm(t)
    result = -1 * delta * sum([dsqu(t)*math.exp(logI(t,theta,data)) for t in range(size,time)])
    return np.asarray(np.reshape(result,[theta.size,theta.size]))

  def logL_hess_p(self,theta, p, *args):
    hessian = np.asmatrix(logL_hess(theta,*args))
    p = np.asmatrix(p)
    return np.asarray(p * hessian )
"""
