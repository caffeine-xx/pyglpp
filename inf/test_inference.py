import numpy as np
from scipy import *

import math as m

import basis as b
import likelihood_model as lm
import trains as tr

reload(tr)
reload(b)
reload(lm)


# basic parameters
neurons = 2
stimuli = 2
delta = 0.1
time = np.arange(0.1, 10.0, delta)
tau = np.arange(0.1, 2.0, 0.1)

# simulation
stim = np.zeros([stimuli,time.size])+0.5
spikes = np.zeros([neurons,time.size])
sparse = neurons*[[]]

for i in range(0,neurons):
  spikes[i,:] = tr.rand_train(stim[0,:])
  sparse[i] = tr.spike_times(spikes[i,:])

# initialization
basis = b.straight_basis(0.5)
basis = np.array([basis(1,1,tau) for i in range(0,4)])

mn = lm.MultiNeuron(delta,tau,stim,spikes,sparse,basis,basis)

def test_pack_unpack():
  K = np.random.random([neurons,stimuli,4])
  H = np.random.random([neurons,neurons,4])
  Mu = np.random.random([neurons])
  th,ar = mn.pack(K,H,Mu)
  oK,oH,oM = mn.unpack(th,ar)
  assert (K==oK).all() and (H==oH).all() and (Mu==oM).all()

def test_multineuron():
  # init model
  x = np.array([
    [0,1,0,0, 0,0,0,0],
    [0,0,0,0, 2,0,0,0]])
  y = np.array([
    [0,1,0,0, 0,0,0,0],
    [0,0,0,0, 1,0,0,0]])
  b = np.array([
    [1,1],
    [2,1]])
  s = [[1],[4]]
  t = np.array([0,1])
  mn = lm.MultiNeuron(b,b)
  mn.set_data(1,t,x,s)

  # check bases
  bx =  np.array([[[ 1.0,  1.0],
          [ 1.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 2.0,  2.0],
          [ 2.0,  4.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]]])
  by = np.array([[[ 1.0,  1.0],
          [ 1.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 1.0,  1.0],
          [ 1.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]]])

  assert (bx==mn.base_stims).all()
  assert (by==mn.base_spikes).all()

  # check intensities
  K = np.ones((2, 2, 2))
  K[1,:,:] = K[1,:,:] * 2
  H = np.ones((2, 2, 2))
  H[1,:,:] = H[1,:,:] * 2
  M = np.zeros(2)
  I = mn.logI(K,H,M)
  vI = np.array([[4,6,0,6, 9,0,0,0],
                 [8,12,0,12, 18,0,0,0]])

  assert (I==vI).all()

  # check likelihoods
  l0 = vI[0,1] - np.sum(np.exp(vI[0,:]))
  l1 = vI[1,4] - np.sum(np.exp(vI[1,:]))
  L  = l0 + l1
  cL = mn.logL(K,H,M)

  assert (L - mn.logL(K,H,M)) < 0.001

  # check gradients
  eI = np.exp(I)
  dK, dH, dM = mn.logL_grad(K,H,M)
  for i in range(0,2):
    for j in range(0,2):
      for l in range(0,2):
        sp = s[i][0]
        g = bx[j,sp,l] - np.sum(bx[j,:,l] * eI[i,:])
        assert abs(g-dK[i,j,l]) < 0.01
        g = by[j,sp,l] - np.sum(by[j,:,l] * eI[i,:])
        assert abs(g-dH[i,j,l]) < 0.01
        


class LLStub(lm.LikelihoodModel):
  def logL(self,x,n,c):
    return -1*(x**n)+c

  def logL_grad(self,x,n,c):
    return -1*n*(x**(n-1)),0,0

  def pack(self,x,n,c):
    theta=x
    args=(n,c)
    return theta,args

  def unpack(self,theta,args):
    x=theta
    (n,c)=args
    return x,n,c

def test_mlestimator():

  lls = LLStub()
  c = 5
  n = 2
  x0 = 10

  est = mle.MLEstimator(lls)
  (x,n,c) = est.maximize(x0,n,c)
  
  assert x<0.01
