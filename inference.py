from utils import print_timing
import sys
import math as m
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as opt
import scipy.stats as st
from memoize import Memoize
from signals import *

def run_bases(bases, data):
  """ Correlates a dataset with a set of bases.
      Takes a 2D array to a 3D array """
  rows,cols = data.shape
  num,size = bases.shape
  result = np.zeros([rows,cols,num])
  for i in range(0,num):
    result[:,:,i] = run_filter(bases[i],data)
  return result

def run_filter(filt, data):
  orig = -1*int(np.floor(filt.size/2))
  return nd.convolve1d(data, filt, mode='constant', cval=0.0, origin=orig)

class LikelihoodModel:
  """ A probabilistic model for which we can calculate likelihood """
  
  def set_data(self, *args):
    raise Exception('not implemented')

  def logL(self,*args):
    raise Exception('not implemented')

  def logL_grad(self,*args):
    raise Exception('not implemented')

  def unpack(self,theta, args):
    raise Exception('not implemented')

  def pack(self,*a):
    raise Exception('not implemented')

  def random_args(self):
    raise Exception('not implemented')

  def zero_args(self):
    raise Exception('not implemented')

  def sparse_to_spikes(self, sparse):
    ''' Utility method - converts sparse representations of
        spike trains to full representation '''
    for i,s in enumerate(sparse):
      self.spikes[i,s]=1



default_basis_length = Trial(0,2*pi,dt=pi/16)
default_spike_basis  = SineBasisGenerator(7,dim=5).generate(default_basis_length)
default_stim_basis   = SineBasisGenerator(7,dim=5).generate(default_basis_length)

class MultiNeuron(LikelihoodModel):
  """ Multi-neuron Poisson model with user-specified bases"""
  def __init__(self, stim_basis=default_stim_basis.signal, 
                     spike_basis=default_spike_basis.signal):
    self.stim_basis  = stim_basis
    self.spike_basis = spike_basis
    self.spike_b     = spike_basis.shape[0]
    self.stim_b      = stim_basis.shape[0]

  def set_data(self, timestep, duration, stims, sparse):
    # dimensions
    self.delta   = timestep
    self.T       = duration
    self.N       = len(sparse)
    self.Nx      = stims.shape[0]
    # data
    self.sparse  = sparse
    self.spikes  = np.zeros((self.N,self.T),dtype='float64')
    self.stims   = stims
    self.sparse_to_spikes(sparse)
    # basis
    self.base_spikes = run_bases(self.spike_basis, self.spikes)
    self.base_stims  = run_bases(self.stim_basis, self.stims)

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
  
  def random_args(self):
    Ksize = (self.N, self.Nx, self.stim_b)
    Hsize = (self.N, self.N, self.spike_b)
    Msize = (self.N)
    return (0.01*np.random.standard_normal(Ksize), 0.01*np.random.standard_normal(Hsize), 0.01*np.random.standard_normal(Msize))

  def zero_args(self):
    Ksize = (self.N, self.Nx, self.stim_b)
    Hsize = (self.N, self.N, self.spike_b)
    Msize = (self.N)
    return (np.zeros(Ksize), np.zeros(Hsize), np.zeros(Msize))

  def logI(self, K, H, Mu):
    I = np.zeros([self.N,self.T],dtype='float64')
    for i in xrange(self.N):
      for j in xrange(self.Nx):
        I[i,:] += np.dot(self.base_stims[j,:,:],K[i,j,:]).T
      for j in xrange(self.N):
        I[i,:] += np.dot(self.base_spikes[j,:,:],H[i,j,:]).T
      I[i,:] += Mu[i]
    return I

  def logL(self, K, H, Mu):
    I = self.logI(K,H,Mu)
    t1 = 0
    for i in xrange(self.N):
      t1 += np.sum(I[i,self.sparse[i]])
    t2 = np.sum(np.sum(np.ma.exp(I)))
    return t1 - self.delta*t2

  @print_timing
  def logL_grad(self, K, H, Mu):
    I = self.logI(K,H,Mu)
    expI = np.ma.exp(I)
    dK = np.zeros([self.N, self.Nx, self.stim_b],dtype='float64')
    dH = np.zeros([self.N, self.N, self.spike_b],dtype='float64')
    dM = np.zeros([self.N],dtype='float64')
    for i in xrange(self.N):
      for j in xrange(self.Nx):
        dK[i,j,:] = np.sum(self.base_stims[j,self.sparse[i],:],0)
        dK[i,j,:] -= self.delta * np.sum(self.base_stims[j,:,:] * expI[i,:].reshape((I[i,:].size,1)),0)
      for j in xrange(self.N):
        dH[i,j,:] = np.sum(self.base_spikes[j,self.sparse[i],:],0)
        dH[i,j,:] -= self.delta * np.sum(self.base_spikes[j,:,:] * expI[i,:].reshape((I[i,:].size,1)),0)
      dM[i]= len(self.sparse[i])-self.delta*np.sum(expI[i,:])
    return dK, dH, dM

  @print_timing
  def logL_hess_p(self, K, H, Mu, K1, H1, Mu1):
    ''' Calculates Hessian matrix of the log-likelihood function, 
        to be matrix multiplied with the PACKED parameter vector. '''
    expI = np.ma.exp(self.logI(K,H,Mu))

    # Temp variables for easy naming
    Xf     = self.base_stims
    Yf     = self.base_spikes

    # Basis-filtered data * intensity, per neuron
    IK  = -self.delta * np.array([expI[i,:].reshape((1,self.T,1))*Xf for i in xrange(self.N)])
    IH  = -self.delta * np.array([expI[i,:].reshape((1,self.T,1))*Yf for i in xrange(self.N)])
    IMM = -self.delta * np.sum(expI,axis=1)

    # Non-zero blocks of Hessian matrix
    IKK = np.tensordot(IK,Xf,axes=[2,1])
    IKH = np.tensordot(IK,Yf,axes=[2,1])
    IHH = np.tensordot(IH,Yf,axes=[2,1])
    IKM = np.sum(IK,axis=2)
    IHM = np.sum(IH,axis=2)

    # Products of Hessian with Parameters
    PK = np.zeros(K.shape)
    PH = np.zeros(H.shape)
    PM = np.zeros(Mu.shape)
    ax = [(2,3),(0,1)]
    td = np.tensordot
    for i in xrange(self.N):
      PK[i] = td(IKK[i],K1[i],axes=ax)+td(IKH[i],H1[i],axes=ax)+IKM[i]*Mu1[i]
      PH[i] = td(IKH[i],K1[i],axes=[(0,1),(0,1)])+td(IHH[i],H1[i],axes=ax)+IHM[i]*Mu1[i]
      PM[i] = (IKM[i]*K1[i]).sum()+(IHM[i]*H1[i]).sum()+IMM[i]*Mu1[i]
    return PK,PH,PM

class FixedConnections(MultiNeuron):
  ''' Neuron model allowing one to fix connections between specific pairs
  of neurons to certain values. '''

  def __init__(self, stim_basis, spike_basis):
    MultiNeuron.__init__(self, stim_basis, spike_basis)
  
  def set_connectivity(self, conn):
    ''' Takes a list of lists: 
        conn[i] = [... j ...] if i=>j exists. '''
    self.conn = conn

  def logI(self, K, H, Mu):
    I = np.zeros([self.N,self.T],dtype='float64')
    for i in xrange(self.N):
      for j in xrange(self.Nx):
        res, s  = np.average(self.base_stims[j,:,:],axis=1,weights=K[i,j,:],returned=True)
        I[i,:] += res*s
      for j in self.conn[i]:
        res, s  = np.average(self.base_spikes[j,:,:],axis=1,weights=H[i,j,:],returned=True)
        I[i,:] += res*s
      I[i,:] += Mu[i]
    return I

  def logL(self, K, H, Mu):
    I = self.logI(K,H,Mu)
    t1 = 0
    for i in xrange(self.N):
      t1 += np.sum(I[i,self.sparse[i]])
    t2 = np.sum(np.sum(np.ma.exp(I)))
    return t1 - self.delta*t2

  def logL_grad(self, K, H, Mu):
    I = self.logI(K,H,Mu)
    expI = np.ma.exp(I)
    dK = np.zeros([self.N, self.Nx, self.stim_b],dtype='float64')
    dH = np.zeros([self.N, self.N, self.spike_b],dtype='float64')
    dM = np.zeros([self.N],dtype='float64')
    for i in xrange(self.N):
      for j in xrange(self.Nx):
        dK[i,j,:] = np.sum(self.base_stims[j,self.sparse[i],:],0)
        dK[i,j,:] -= self.delta * np.sum(self.base_stims[j,:,:] * expI[i,:].reshape((I[i,:].size,1)),0)
      for j in self.conn[i]:
        dH[i,j,:] = np.sum(self.base_spikes[j,self.sparse[i],:],0)
        dH[i,j,:] -= self.delta * np.sum(self.base_spikes[j,:,:] * expI[i,:].reshape((I[i,:].size,1)),0)
      dM[i]= len(self.sparse[i])-self.delta*np.sum(expI[i,:])
    return dK, dH, dM

class SimpleModel(MultiNeuron):
  ''' Model using Signal api: dirty hack wrapper, should
    write a cleaner version. '''
  def __init__(self, in_filter=default_stim_basis, 
                     out_filter=default_spike_basis):
    self.in_filter = in_filter
    self.out_filter = out_filter
    MultiNeuron.__init__(self, in_filter.signal, out_filter.signal)

  def set_data(self, trial, in_signal, out_signal):
    ''' Uses the Signals api.  Parameters:
          - trial: Time dimension of experiment
          - in_signal: Sparse or dense signal of inputs
          - out_signal: Sparse or dense signal of outputs (spikes) '''
    MultiNeuron.set_data(self, 
      trial.dt, trial.length(),
      in_signal.signal, out_signal.sparse_bins())

class MLEstimator(LikelihoodModel):
  """ Decorator for LikelihoodModels
      Allows one to perform maximum likelihood inference on any
      likelihood model (i.e. one that exposes logL and logL_grad,
      and pack & unpack) """
  iters = 0
  def __init__(self, model):
    self.model = model

  def logL(self,theta, *args):
    a = self.model.unpack(theta, args)
    return -1.0*self.model.logL(*a)

  def logL_grad(self,theta, *args):
    a = self.model.unpack(theta, args)
    theta, shape = self.model.pack(*tuple(self.model.logL_grad(*a)))
    return -1.0*theta

  def logL_hess_p(self, theta, p, *args):
    a = self.model.unpack(theta, args)
    p = self.model.unpack(p, args)
    return -1.0*self.model.pack(*tuple(self.model.logL_hess_p(*tuple(a+p))))[0]
  
  @print_timing
  def maximize_cg(self,*a):
    ''' Doesn't use Hessian matrix - used for checking the Hessian version '''
    self.iters=0
    theta, args = self.model.pack(*a)
    theta = opt.fmin_cg(self.logL, theta, self.logL_grad,  args=args, maxiter=500, gtol=1.0e-02, callback=self.callback)
    return self.model.unpack(theta, args)

  @print_timing
  def maximize(self,*a):
    self.iters=0
    theta, args = self.model.pack(*a)
    theta = opt.fmin_ncg(self.logL, theta, self.logL_grad, fhess_p=self.logL_hess_p, args=args, callback=self.callback,maxiter=20)
    self.iters=0
    theta2 = opt.fmin_cg(self.logL, theta, self.logL_grad, args=args, maxiter=500, gtol=1.0e-02, callback=self.callback)
    return self.model.unpack(theta, args)

  def callback(self,x):
    print self.iters,
    self.iters += 1
    sys.stdout.flush()
