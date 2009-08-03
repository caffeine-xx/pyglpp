import numpy.random as ra
from brian import *
from brian.library.IF import *
from numpy import *
from signals import *
from result import *
from inference import run_bases
from utils import print_timing

@vectorize
def poisson(lam):
  return ra.poisson(max(0,lam),1)>0

class LNPSimulator:
  ''' Simulates a set of LNPoisson neurons given a 
      set of parameters '''

  def __init__(self, spike_basis, stim_basis, N, K, H, Mu):
    self.K,self.H,self.Mu = K,H,Mu
    self.spike_basis = atleast_2d(spike_basis)
    self.tau_sp      = self.spike_basis.shape[1]
    self.stim_basis  = atleast_2d(stim_basis)
    self.tau_st      = self.stim_basis.shape[1]
    self.N           = N
    self. params = {
      'K': K, 'H':H, 'Mu':Mu, 'N':N,
      'stim_basis':stim_basis,
      'spike_basis':spike_basis
    } 

  def logI(self, K, H, Mu, bsp, bst):
    ''' Calculates log intensity for a single value '''
    I = zeros(self.N,dtype='float64')
    for i in xrange(self.N):
      for j,dat in enumerate(bst):
        I[i] += dot(dat,K[i,j,:]) 
      for j,dat in enumerate(bsp):
        I[i] += dot(dat,H[i,j,:]) 
      I[i] += Mu[i]
    return I
  
  def rebase(self, basis, data):
    ''' Run bases only for the latest value of signal'''
    num,tau = basis.shape
    row,col = data.shape
    result  = dot(data[:,-tau:],basis[:,-min(tau,col):].T)
    return result

  @print_timing
  def run(self, signal):

    (K,H,Mu) = (self.K,self.H,self.Mu)
    trial  = signal.trial
    stims  = signal.signal
    self.Nx= signal.dims()
    bst    = run_bases(self.stim_basis, stims)
    spikes = zeros((self.N,trial.length()))
    lams   = zeros((self.N,trial.length()))
    raster = []
    flipsp = fliplr(self.spike_basis)

    for t in xrange(1,trial.length()):
      bsp         = self.rebase(flipsp,spikes[:,:t])
      lams[:,t]   = self.logI(K,H,Mu,bsp,bst[:,t,:])
      spikes[:,t] = poisson(exp(lams[:,t]))
      spiked      = flatnonzero(spikes[:,t])
      raster      = raster + zip(spiked, [trial.bin_to_time(t)]*len(spiked))

    return Result(self.params, signal, [], raster, {'lambda': (lams)})

class Simulator:
  ''' Wrapper for the Brian simulator '''

  # Izhikevich equations with synapses
  eqs = ''' dv/dt  = ((0.04/mV)*v**2 + 5.0*v + 140.0*mV - w)/ms + I/C_m : volt
            dw/dt  = a*(b*v - w)/ms : volt
            I      = ge*(Ee-v) - gi*(Ei-v) : amp
            dge/dt = -ge / te : siemens
            dgi/dt = -gi / ti : siemens'''

  # Default simulation parameters
  p = {
    'model':  {'a':0.02,        'b':0.2, 
               'Ee':0*mV,       'Ei':-10.0*mV, 
               'te':10*ms,      'ti':9*ms,
               'C_m':21.0*pF                 },
    'init':   {'gei':0.0*nS,     'gii':0.0*nS,
               'vm':-65.0*mV,    'w':-18.0*mV,
                'I': 0.0*amp                 },
    'neurons':{'N':40,         'Ni': 8,
               'threshold':     'v > 30.0*mV',
               'reset':         'v = -65.0*mV;'+
                                'w += 8.0*mV'},
    'connect':{'weight':1.0*nS, 'sparseness':0.10,
               'state':'ge'                 },
    'inhibit':{'weight':1.0*nS, 'sparseness':0.10,
               'state':'gi'                 },
    'inputs': {'state':'ge',    'weight':1.0*nS, 
               'delay':1*ms,    'sparseness':0.10},
    'record': {'v': False,      'I':False,
               'ge': False,     'gi': False }
  }

  def __init__(self, **p):
    ''' Parameters:
          - model -> DiffEq parameters for the model
          - init  -> Initialization of the neuron group
          - neurons -> Number of neurons
          - connect -> Interior connection properties
          - inhibit -> Inhibitory connection properties 
          - inputs  -> Connectivity of inputs to neurons 
          - record  -> Which state variables to record (spikes are
                        always recorde)'''
    for k in p: self.p[k].update(p[k])
    Ni = self.p['neurons']['Ni']

    self.model   = Equations(self.eqs, **self.p['model'])
    self.neurons = NeuronGroup(model=self.model,**self.p['neurons'])
    self.connect = Connection(self.neurons[Ni:], self.neurons, **self.p['connect'])

    for k in self.p['init']:
      setattr(self.neurons, k, self.p['init'][k])

    if Ni > 0: 
      self.inhibit = Connection(self.neurons[:Ni], self.neurons, **self.p['inhibit'])
    else: self.inhibit = None

  @print_timing
  def run(self, signal=GaussianNoiseGenerator(10.0, 2.0, 10).generate(Trial(0.0,2.0,0.001)),
                input_param={}):
    ''' Runs a trial of the simulation using the input signal provided '''
    self.p['inputs'].update(input_param)
    trial       = signal.trial
    clock       = self.trial_to_clock(trial)

    input       = PoissonGroup(signal.dims(), rates=lambda t: signal[t]*Hz)
    in_conn     = Connection(input, self.neurons, **self.p['inputs'])
    in_monitor  = SpikeMonitor(input, record=True)
    out_monitor = SpikeMonitor(self.neurons, record=True)

    self.neurons.clock=clock
    input.clock=clock

    network     = Network(self.neurons, self.connect, input,
                          in_conn, in_monitor, out_monitor)

    if self.inhibit: network.add(self.inhibit)

    monitors    = self.add_monitors(network)
    network.run(trial.duration()*second)
    values      = self.get_values(monitors)

    return Result(self.p, signal, in_monitor.spikes, out_monitor.spikes, values)

  def add_monitors(self, network):
    monitors = {}
    for k in self.p['record']:
      if self.p['record'][k]:
        monitors[k] = StateMonitor(self.neurons, k, record=True)
        network.add(monitors[k])
    return monitors
  
  def trial_to_clock(self, trial):
    clock = Clock(dt = trial.dt * second, t = trial.t_start * second)
    return clock

  def get_values(self, monitors):
    values = {}
    for k in monitors:
      values[k] = monitors[k].values
    return values

if(__name__=="__main__"):
  import sys
  filename = "results/"+sys.argv[1]
  print "===== Simulator: "
  sim = Simulator()
  print " >> parameters: "
  print sim.p
  print " >> running..."
  res = sim.run()
  print " >> writing output to %s" % filename
  save_result(filename, res)

