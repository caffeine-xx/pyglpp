import numpy as np
import math as m

from scipy import *

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

def test_logI():

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

  mn = lm.MultiNeuron(1,t,x,y,s,b,b)

  bx =  [[[ 1.0,  1.0],
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
          [ 0.0,  0.0]]]
  by =  [[[ 1.0,  1.0],
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
          [ 0.0,  0.0]]]
  assert (bx==mn.base_stims).all()
  assert (by==mn.base_spikes).all()

  K = np.ones((2, 2, 2))
  K[1,:,:] = K[1,:,:] * 2
  H = np.ones((2, 2, 2))
  H[1,:,:] = H[1,:,:] * 2
  M = np.zeros(2)

  I = mn.logI(K,H,M)
  vI = np.array([[4,6,0,6, 9,0,0,0],
                 [8,12,0,12, 18,0,0,0]])
  assert (I==vI).all()


