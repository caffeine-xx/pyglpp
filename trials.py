from numpy import *
from brian import *
from brian.library.IF import *
import cPickle

from signals import *

def two_neuron(prefix, perm_input=0, perm_weight=0):
  neur = {'N': 2,    'Ni': 0}
  reco = {'v': True, 'I': True}
  
  (weight, inputs) = weight_permutations(perm_weight, perm_input)
  
  conn = {'weight':weight}
  inpu = {'weight':inputs}
  
  time = Trial(t_stop=1.0)
  sign = GaussianNoiseGenerator(10.0, 2.0, 1).generate(time)
  return ({'neurons': neur, 'record':reco, 
           'connect': conn, 'inputs':inpu},sign) 

def one_neuron(prefix,*args):
  neur = {'N': 1, 'Ni': 0}
  conn = {'weight':0.0*nS}
  inhi = {'weight':0.0*nS}
  
  time = Trial()
  sign = GaussianNoiseGenerator(10.0, 2.0, 1).generate(time)
  
  return ({'neurons': neur, 'inhibit':inhi, 
            'connect': conn},sign) 

def random_net(prefix,N=200,lam=1.0, pi=0.1, rs=0.02):
  N = int(N)
  lam,pi,rs = map(float, (lam, pi,rs))

  weight = randomly_switch(taur_connector(N,N,lam),rs)*(0.8+random.rand(N,N))*nS
  print "mean conn strength:",weight.sum(0).mean()

  from matplotlib import pylab
  pylab.matshow(weight)

  neur = {'N':N, 'Ni':int(0.2 * N)}
  conn = {'weight':weight, 'delay':True, 'max_delay':10.0*ms}
  inpu = {'sparseness': pi}
  inhi = {'sparseness': 0.05}

  time = Trial(0.0,20.0,0.001)
  sign = GaussianNoiseGenerator(4.8,1.0,10).generate(time)
  return ({'neurons':neur, 'connect': conn, 'inhibit':inhi, 'inputs':inpu}, sign)

def randomly_switch(W,p=0.01):
  for i in xrange(W.shape[0]):
    for j in xrange(W.shape[1]):
      if (i != j) and rand()<p: W[i,j] = (W[i,j]+1)%2
  return W

def taur_connector(P, Q, la=2.0):
  ''' Connector with probability related to inverted square distance of neurons
      (where 'distance' is a made-up concept based on ID numbers).'''
  W = zeros((P,Q))
  for i in xrange(P):
    for j in xrange(Q):
      if not (i==j): W[i,j] = rand() < taur_dist(i,j,la)
  return W

def taur_dist(i,j,la):
  return exp(-1*sqrt((i-j)**2)/la)

def weight_permutations(perm_weight, perm_input):
  weight = [array([[1.0, 0.0],[0.0, 0.0]])*nS,
            array([[0.0, 1.0],[0.0, 0.0]])*nS,
            array([[0.0, 0.0],[1.0, 0.0]])*nS,
            array([[0.0, 0.0],[0.0, 1.0]])*nS]
  
  inputs = [array([0.0, 0.0])*nS,
            array([1.0, 0.0])*nS,
            array([1.0, 1.0])*nS]
  inputs = map(atleast_2d, inputs)
  return (weight[perm_weight],inputs[perm_input])

def set_connection_voltage(conn, weights):
  for i in xrange(len(weights[:,0])):
    for j in xrange(len(weights[0,:])):
      conn[i,j] = weights[i,j]
  return conn


