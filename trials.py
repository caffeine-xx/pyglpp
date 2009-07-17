from numpy import *
from brian import *
from brian.library.IF import *
import cPickle

def two_neuron(prefix, perm_input=0, perm_weight=0):
  neur = {'N': 2, 'Ni': 0}
  reco = {'v': True, 'I': True}
  
  (weight, inputs) = weight_permutations(perm_weight, perm_input)
  
  conn = {'weight':weight}
  inpu = {'weight':inputs}
  
  time = Trial(t_stop=1.0)
  sign = GaussianNoiseGenerator(30.0, 4.0, 1).generate(time)
  return ({'neurons': neur, 'record':reco, 
           'connect': conn, 'input':input},sign) 

def one_neuron(prefix,*args):
  neur = {'N': 1, 'Ni': 0}
  conn = {'weight':0.0*nS}
  inhi = {'weight':0.0*nS}
  
  time = Trial()
  sign = GaussianNoiseGenerator(50.0, 30.0, 1).generate(time)
  
  return ({'neurons': neur, 'inhibit':inhi, 
            'connect': conn},sign) 

def weight_permutations(perm_weight, perm_input):
  weight = [array([[1.0, 0.0],[0.0, 0.0]])*nS,
            array([[0.0, 1.0],[0.0, 0.0]])*nS,
            array([[0.0, 0.0],[1.0, 0.0]])*nS,
            array([[0.0, 0.0],[0.0, 1.0]])*nS]
  
  inputs = [array([0.0, 0.0])*nS,
            array([1.0, 0.0])*nS,
            array([1.0, 1.0])*nS]
  
  return (weight[perm_weight],input[perm_input])


def d2_connector(N, inhib=[], weight=1.0):
  ''' Connector with probability related to inverted square distance of neurons
      (where 'distance' is a made-up concept based on ID numbers).
      The inhibitory argument lists neuron IDs that are inhibitory, so that
      their weights to other neurons should be negative. 
      Units: [neurons x neurons] * nS '''
  W = zeros((N,N))
  for i in xrange(N):
    for j in xrange(N):
      W[i,j] = rand() < weight*(1.0 - (float((i-j)%N)/float(N))**2)
  W[inhib,:] = -1 * W[inhib,:]
  return W

def set_connection_voltage(conn, weights):
  for i in xrange(len(weights[:,0])):
    for j in xrange(len(weights[0,:])):
      conn[i,j] = weights[i,j]
  return conn


