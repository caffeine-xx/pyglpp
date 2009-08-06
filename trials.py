from matplotlib import pylab
from numpy import *
from brian import *
from brian.library.IF import *
import cPickle

import simulator as s
from signals import *
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

def two_izh(t=1.0):
  t = float(t)
  
  model= {'C_m': 4.0*pF}
  neur = {'N': 2,    'Ni': 0}
  reco = {'v': True, 'I': False}
  
  H = array([[0.0, 0.0],[1.0, 0.0]])*2*nS 
  K = array([[1.0, 0.0],[0.0, 1.0]])*2*nS

  inpu = {'weight':K, 'sparseness':None}
  conn = {'weight':H, 'delay':True, 'max_delay':10*ms}

  trial = Trial(t_start=0.0, t_stop=t, dt=0.001)
  sign  = Signal(trial, array([gaussian_wave(trial, 15*t)*10.0, gaussian_wave(trial, 15*t)*10.0]))

  simu = s.Simulator(neurons= neur, record=reco, 
           connect= conn, inputs=inpu,
           model=model)

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
  sign = GaussianNoiseGenerator(20.0,1.0,10).generate(time)
  simu = s.Simulator(neurons=neur, connect=conn, inhibit=inhi, inputs=inpu)
  print simu.p
  resu = simu.run(signal)
  return resu

# --- LNP

def two_lnp_rand(t=1.0):
  ''' Two neurons, two stimuli, no connection, 5ms delay input '''
  t = float(t)
  Yb = array([[0.0, 0.0, 0.0, 0.0, 1.0]])
  Xb = Yb
  K = array([[[5.0], [0.0]], [[0.0],[5.0]]])
  H = array([[[0.0], [0.0]],[[0.0],[0.0]]])
  M = array([-5.0, -4.8])
  trial = Trial(t_start=0.0, t_stop=t, dt=0.001)
  signal = Signal(trial, array([gaussian_wave(trial, 40)*0.6,sine_wave(trial, 40)*0.3+0.3]))
  simu = s.LNPSimulator(Yb,Xb,2,K,H,M)
  print simu.params
  return simu.run(signal)

def two_lnp(t=1.0):
  ''' Two neurons, one stimulus, 0->1 connection, 5ms delay input & net'''
  t = float(t)
  dX,dY = 2,2
  Xb = SineBasisGenerator(a=2.7,dim=dX).generate().signal
  Yb = SineBasisGenerator(a=2.7,dim=dY).generate().signal
  #Yb = atleast_2d(ones(8))/3.0#
  #Xb = atleast_2d(zeros(8)); Xb[:,0]=1
  K = array([[[0.8, 0.4], [0.0, 0.0]], 
             [[0.0, 0.0], [0.2, 0.1]]])

  H = array([[[0.0, 0.0], [0.0, 0.0]], 
             [[0.6, 0.3], [0.0, 0.0]]])
  M = array([-4.0, -4.0])
  trial   = Trial(t_start=0.0, t_stop=t, dt=0.001)
  signal = Signal(trial, array([gaussian_wave(trial, 15*t), gaussian_wave(trial, 15*t)]))
  simu = s.LNPSimulator(Yb,Xb,2,K,H,M)
  print simu.params
  return simu.run(signal)

def random_lnp(N=20, t=0.1, lam=2.0,rs=0.02):

  N,t,lam,rs  = int(N),float(t),float(lam),float(rs)
  dX,dY,dS = 5,5,1
  Ni       = int(0.2*N)
  SparseI  = 0.4
  SparseS  = 0.5

  Kvar,Kmean = 0.1, 0.00
  Hvar,Hmean = 0.1, 0.00

  trial  = Trial(t_start=0.0, t_stop=t, dt=0.001)
  signal = SingleSpikeGenerator(dim=dS).generate(trial)
  #signal = GaussianNoiseGenerator(0.01, 0.01, dS).generate(trial)
  
  Xb = SineBasisGenerator(a=2.7,dim=dX).generate().signal
  Yb = SineBasisGenerator(a=2.7,dim=dY).generate().signal

  K = random.randn(N,dS,dX)*Kvar+Kmean
  H = random.randn(N,N,dY)*Hvar+Hmean
  M = random.randn(N)*0.1
  
  weight = randomly_switch(taur_connector(N,N,lam),rs)
  weight[:,:Ni] = (random.rand(N,Ni)<SparseI)*-1.0
  
  for i in range(N): 
    for k in range(dS):
      K[i,k,:] = (random.rand() < SparseS) * K[i,k,:]
    for j in range(N):
      H[i,j,:] = H[i,j,:] + 0.5 * weight[i,j]
    H[i,i,:] = arange(0.0,1.0,1./dS)-0.5

  w2 = average(H,axis=2)
  pylab.matshow(weight)
  pylab.matshow(w2)

  simu = s.LNPSimulator(Yb,Xb,N,K,H,M)
  print simu.params
  resu = simu.run(signal)
  return resu

# --- Stimulation

def sine_wave(trial, periods = 100):
  wave = zeros(trial.length())
  wave = sin(arange(trial.length())*2*pi*periods/trial.length())
  return wave 

def gaussian_wave(trial, inter = 100):
  wave  = zeros(trial.length())
  curr  = 0
  while curr < trial.length():
    t = random.normal(loc=trial.length()/inter, scale=trial.length()/(inter*2)) 
    r = random.rand()
    print r
    wave[curr:min(curr+int(t),trial.length()-1)] = r
    curr = curr + int(t)
  return wave

def regular_spikes(trial, inter=30):
  indices = range(0, trial.length(), inter)
  signal  = zeros((1, trial.length()))
  signal[0,indices] = 1.0
  return signal

# --- Connectivity

def randomly_switch(W,p=0.01):
  for i in xrange(W.shape[0]):
    for j in xrange(W.shape[1]):
      if (i != j) and rand()<p: W[i,j] = (W[i,j]+1)%2
  return W

def weight_permutations(perm_weight, perm_input):

  weight = [array([[0.0, 0.0],[0.0, 0.0]]), #0: no connect
            array([[1.0, 0.0],[0.0, 0.0]]), #1: 0,0
            array([[0.0, 1.0],[0.0, 0.0]]), #2: 0,1
            array([[0.0, 0.0],[1.0, 0.0]]), #3: 1,0
            array([[0.0, 0.0],[0.0, 1.0]]), #4: 1,1
            array([[0.0, 1.0],[1.0, 0.0]])] #5: 0,1 + 1,0

  inputs = [array([0.0, 0.0]), # none
            array([1.0, 0.0]), # 0
            array([1.0, 1.0])] # 0 + 1

  return map(atleast_2d,(weight[perm_weight],inputs[perm_input]))

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

def set_connection_voltage(conn, weights):
  for i in xrange(len(weights[:,0])):
    for j in xrange(len(weights[0,:])):
      conn[i,j] = weights[i,j]
  return conn


