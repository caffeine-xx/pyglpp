from matplotlib import pylab
from numpy import *
from brian import *
from brian.library.IF import *
import cPickle

import simulator as s
from signals import *

def two_neuron(pw=0, pi=0):
  (pw,pi) = map(int, [pw, pi])
  
  model= {'C_m': 4.0*pF}
  neur = {'N': 2,    'Ni': 0}
  reco = {'v': True, 'I': True}
  
  (weight, inputs) = weight_permutations(pw, pi)
  
  conn = {'weight':weight*2, 'delay':True, 'max_delay':10*ms}
  inpu = {'weight':inputs,'sparseness':None}
  
  time = Trial(t_stop=2.0)
  sign = GaussianNoiseGenerator(15.0, 2.0, 1).generate(time)

  simu = s.Simulator(neurons= neur, record=reco, 
           connect= conn, inputs=inpu,
           model=model)
  print simu.p
  resu = simu.run(sign) 
  return resu

def one_neuron(t_stop=1.0):
  t_stop = float(t_stop)

  neur = {'N': 1, 'Ni': 0}
  conn = {'weight':0.0*nS}
  inhi = {'weight':0.0*nS}
  inpu = {'weight':array([[1.5*nS]]),'sparseness':None}
  mode = {'C_m': 2.0*pF}
  reco = {'v': True, 'I': True}

  time = Trial(t_stop=t_stop)
  sign = GaussianNoiseGenerator(20.0).generate(time)
  
  simu = s.Simulator(neurons= neur, inhibit=inhi, 
            connect= conn, model= mode,
            record= reco, inputs= inpu)
  print simu.p
  resu = simu.run(sign) 
  return resu

def random_net(N=200,lam=1.0, pi=0.1, rs=0.02, inh=0.2):
  N = int(N)
  lam,pi,rs,inh = map(float, (lam, pi,rs,inh))
  Ni = int(0.2 * N)
  weight = randomly_switch(taur_connector(N-Ni,N-Ni,lam),rs)*(0.8+random.rand(N-Ni,N-Ni))*nS

  print "  Mean synapses per neuron:",weight.sum(0).mean()

  neur = {'N':N, 'Ni':int(0.2 * N)}
  conn = {'weight':weight, 'delay':True, 'max_delay':10.0*ms}
  inpu = {'sparseness': pi}
  inhi = {'sparseness': inh}

  time = Trial(0.0,0.1,0.001)
  sign = GaussianNoiseGenerator(55.0,1.0,10).generate(time)
  simu = s.Simulator(neurons=neur, connect=conn, inhibit=inhi, inputs=inpu)
  print simu.p
  resu = simu.run(signal)
  return resu

def random_lnp(N=20, t=0.1, lam=2.0,rs=0.02):

  N,t,lam,rs  = int(N),float(t),float(lam),float(rs)
  dX,dY,dS = 5,5,5
  Ni       = int(0.2*N)
  Sparse   = 0.2

  trial  = Trial(t_start=0.0, t_stop=t, dt=0.001)
  signal = GaussianNoiseGenerator(15.0, 4.0, dS).generate(trial)
  
  Xb = SineBasisGenerator(dim=dX).generate().signal
  Yb = SineBasisGenerator(dim=dY).generate().signal
  
  K = zeros((N,dS,dX))
  H = zeros((N,N,dY))
  M = random.randn(N)*2+10
  
  weight = randomly_switch(taur_connector(N,N,lam),rs)

  for i in xrange(N):
    for j in xrange(dS):
      syn = array([10.0+random.randn()*2, -5.0+random.randn(), 1.0+random.randn()*0.3, 1.0+random.randn()*0.3, 1.0+random.randn()*0.3])
      if (random.rand()<Sparse): K[i,j,:] = syn
    for j in xrange(Ni):
      H[i,j,:] = (random.rand()<Sparse) * (-1.0*syn)
    for j in xrange(Ni,N):
      syn = array([10.0+random.randn()*2, -5.0+random.randn(), 1.0+random.randn()*0.3, 1.0+random.randn()*0.3, 1.0+random.randn()*0.3])
      H[i,j,:] = weight[i,j]*syn  
  
  w2 = average(H,axis=2)
  pylab.matshow(weight)
  pylab.matshow(w2)

  simu = s.LNPSimulator(Yb,Xb,N,K,H,M)
  print simu.params
  resu = simu.run(signal)
  return resu

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
            array([[0.0, 0.0],[0.0, 1.0]])*nS,
            array([[0.0, 1.0],[1.0, 0.0]])*nS]

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


