import math as m
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as opt
from memoize import Memoize

def cos_basis(n=4, t=100, a=7, c=1.0, to=2.0):
  phi = lambda n,j: j * m.pi / (2)
  dis = lambda t: a * m.log(t + c)
  bas = lambda n,j,t: (dis(t) > (phi(n,j) - m.pi) and dis(t) < (phi(n,j) + m.pi)) * ((1.0/2.0)*(1 + m.cos(dis(t) - phi(n,j))))
  bas = np.vectorize(bas)
  res = np.zeros((n,t),dtype='float64')
  for j in xrange(n):
    res[j,:] = bas(n,j,np.arange(0.0, to, to/t))
  return res

def straight_basis(a):
  bas = lambda n, j, t: a
  return np.vectorize(bas)

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

class SimpleModel(LikelihoodModel):
  ''' Model using Signal api '''
  def __init__(self, trial, in_signal, in_filter, out_signal, out_filter):
    ''' Uses the Signals api.  Parameters:
          - trial: Time dimension of experiment
          - in_signal: Sparse or dense signal of inputs
          - in_filter: Basis to filter the inputs
          - out_signal: Sparse or dense signal of outputs (spikes)
          - out_filter: Basis to filter the outputs '''
    self.trial = trial
    self.inp   = in_signal
    self.out   = out_signal
    self.in_f  = in_signal.filter_basis(in_filter)
    self.out_f = out_signal.filter_basis(out_filter)

  def simplify(self):
    return (inp, Nx, out, N)

  def logI(self, K, H, Mu):
    I   = np.zeros([self.N,self.T],dtype='float64')
    inp = self.in_f();  Nx = self.inp.dims()
    out = self.out_f(); N  = self.out.dims()
    for i in xrange(N):
      for j in xrange(Nx):
        res, s  = np.average(inp[j,:,:],axis=1,weights=K[i,j,:],returned=True)
        I[i,:] += res*s
      for j in xrange(N):
        res, s  = np.average(out[j,:,:],axis=1,weights=H[i,j,:],returned=True)
        I[i,:] += res*s
      I[i,:] += Mu[i]
    return I



class MultiNeuron(LikelihoodModel):
  """ Multi-neuron Poisson model with user-specified bases"""

  def __init__(self, stim_basis, spike_basis):
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
    return (np.random.random(Ksize), np.random.random(Hsize), np.random.random(Msize))

  def zero_args(self):
    Ksize = (self.N, self.Nx, self.stim_b)
    Hsize = (self.N, self.N, self.spike_b)
    Msize = (self.N)
    return (np.zeros(Ksize), np.zeros(Hsize), np.zeros(Msize))

  def logI(self, K, H, Mu):
    I = np.zeros([self.N,self.T],dtype='float64')
    for i in xrange(self.N):
      for j in xrange(self.Nx):
        res, s  = np.average(self.base_stims[j,:,:],axis=1,weights=K[i,j,:],returned=True)
        I[i,:] += res*s
      for j in xrange(self.N):
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
      for j in xrange(self.N):
        dH[i,j,:] = np.sum(self.base_spikes[j,self.sparse[i],:],0)
        dH[i,j,:] -= self.delta * np.sum(self.base_spikes[j,:,:] * expI[i,:].reshape((I[i,:].size,1)),0)
      dM[i]= len(self.sparse[i])-self.delta*np.sum(expI[i,:])
    return dK, dH, dM

class MLEstimator(LikelihoodModel):
  """ Decorator for LikelihoodModels
      Allows one to perform maximum likelihood inference on any
      likelihood model (i.e. one that exposes logL and logL_grad,
      and pack & unpack) """

  def __init__(self, model):
    self.model = model

  def logL(self,theta, *args):
    a = self.model.unpack(theta, args)
    return -1.0*self.model.logL(*a)

  def logL_grad(self,theta, *args):
    a = self.model.unpack(theta, args)
    theta, shape = self.model.pack(*tuple(self.model.logL_grad(*a)))
    return -1.0*theta

  def maximize(self,*a):
    theta, args = self.model.pack(*a)
    theta = opt.fmin_cg(self.logL, theta, self.logL_grad,  args=args, maxiter=1000, gtol=1.0e-03)
    return self.model.unpack(theta, args)
