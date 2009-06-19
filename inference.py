import math as m
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as opt
from memoize import Memoize

def cos_basis(a=7, c=1.0):
  phi = lambda n,j: j * m.pi / (2)
  dis = lambda t: a * m.log(t + c)
  bas = lambda n,j,t: (dis(t) > (phi(n,j) - m.pi) and dis(t) < (phi(n,j) + m.pi)) * ((1.0/2.0)*(1 + m.cos(dis(t) - phi(n,j))))
  return np.vectorize(bas)

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
  filt = np.atleast_2d(filt)
  data = np.atleast_2d(data)
  orig = -1 * m.floor(filt.size/2)
  return nd.correlate(data, filt, mode='constant', origin=(0,orig))

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

  def sparse_to_spikes(self, sparse):
    ''' Utility method - converts sparse representations of
        spike trains to full representation '''
    for i,s in enumerate(sparse):
      self.spikes[i,s]=1

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
    self.spikes  = np.zeros((self.N,self.T))
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

class MLEstimator(LikelihoodModel):
  """ Decorator for LikelihoodModels
      Allows one to perform maximum likelihood inference on any
      likelihood model (i.e. one that exposes logL and logL_grad,
      and pack & unpack) """

  def __init__(self, model):
    self.model = model

  def logL(self,theta, *args):
    a = self.model.unpack(theta, args)
    return -1*self.model.logL(*a)

  def logL_grad(self,theta, *args):
    a = self.model.unpack(theta, args)
    theta, shape = self.model.pack(*tuple(self.model.logL_grad(*a)))
    return -1*theta

  def maximize(self,*a):
    theta, args = self.model.pack(*a)
    theta = opt.fmin_cg(self.logL, theta, self.logL_grad,  args=args, maxiter=1000)
    return self.model.unpack(theta, args)
