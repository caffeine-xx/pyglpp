import numpy as np
import scipy.optimize as opt
import basis as b
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
  def __init__(self, delta, time, stimulus, spike_trains, sparse, basis):
    self.spikes=spike_trains
    self.stims=stimulus
    self.sparse=sparse
    self.basis=Memoize(basis)
    self.delta=delta
    self.time=time
    self.neu=spike_trains.shape[0]

  def pack(self, K, H, Mu):
    shapes = (K.size, K.shape, H.size, H.shape, Mu.size, Mu.shape)
    theta = np.concatenate(
      np.ndarray.ravel(K),
      np.ndarray.ravel(H), 
      np.ndarray.ravel(Mu))
    return theta, (shapes)
  
  def unpack(self, theta, *args)
    (Ksize, Kshape, Hsize, Hshape, Musize, Mushape) = args[0]
    K = np.ndarray.reshape(theta[0:Ksize], Kshape)
    H = np.ndarray.reshape(theta[Ksize:Hsize], Hshape)
    Mu= np.ndarray.reshape(theta[(Ksize+Hsize):Musize], Mushape)
    return K, H, Mu
  
  @Memoize
  def get_filtered(self, i, K, H, Mu):
    k_i = b.filter_builder(K[i,:,:], self.time, self.basis)
    Fk_i = np.sum(b.run_filter(self.stims, k_i),0)
    h_i = filter_builder(H[i,:,:], self.time, self.basis)
    Fh_i = np.sum(b.run_filter(self.spikes, h_i),0)
    m_i = Mu[i]
    return Fk_i + Fh_i + m_i

  def logL(self, K, H, Mu):
    lam = lambda i: self.get_filtered(self, i, K, H, Mu)
    t1 = 0
    t2 = 0
    for i in range(0,self.neu):
      t1 = t1 + b.selective_sum(lam(i),self.sparse)
      t2 = t2 + np.sum(self.exp(lam(i)))
    return t1 - self.delta*t2

  def logL_grad(self, K, H, Mu)
    lam = lambda i: self.get_filtered(self, i, K, H, Mu)
    


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

