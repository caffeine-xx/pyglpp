from numpy import *
from brian import *
from brian.library.IF import *
from signals import *
import cPickle

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
               'Ee':0*mV,       'Ei':-80.0*mV, 
               'te':10*ms,      'ti':10*ms,
               'C_m':2.0*pF                 }
    'init':   {'gei':10*nS,     'gii':20*nS }
    'neurons':{'N':20,          'Ni': 7,
               'threshold':     'v > 30.0*mV',
               'reset':         'v = -65.0*mV;'+
                                'w += 8.0*mV'}
    'connect':{'weight':0.1*nS, 'sparseness':0.1,
               'state':'ge'                 }
    'inhibit':{'weight':0.1*nS, 'sparseness':0.2,
               'state':'gi'                 },
    'input':  {'state':'ge',    'weight':10*nS, 
               'delay':1*ms                 }
  }

  def __init__(self, **params):
    ''' Parameters:
          - model -> DiffEq parameters for the model
          - init  -> Initialization of the neuron group
          - neurons -> Number of neurons
          - connect -> Interior connection properties
          - inhibit -> Inhibitory connection properties '''
    for k in params: self.p[k].update(params[k])
    Ni = self.p['neurons']['Ni'])
    self.model   = Equations(self.eqs, **self.p['model'])
    self.neurons = NeuronGroup(model=self.model,**self.p['neurons'])
    self.connect = Connection(self.neurons[Ni:], self.neurons, 
      **self.p['connect'])
    for k in self.p['init']: setattr(self.neurons, k, self.p['init'][v])
    if Ni > 0: 
      self.inhibit = Connection(self.neurons[:Ni], self.neurons, **self.p['inhibit'])
    else: self.inhibit = None

  def run(self, signal=GaussianNoiseGenerator(40.0, 10.0, 10).generate(Trial(0.0,100.0,0.1)),
                input_param={}):
    ''' Runs a trial of the simulation using the input signal provided '''
    self.p['input'].update(input_param)

    trial       = signal.trial
    input       = PoissonGroup(signal.dims(), rates=lambda t: signal[t])
    in_conn     = Connection(input, self.neurons, **self.p['input'])
    in_monitor  = SpikeMonitor(input, record=True)
    out_monitor = SpikeMonitor(self.neurons, record=True)
    v_mon       = StateMonitor(self.neurons, 'v', record=True)
    network     = Network(self.neurons, self.connect, input,
                          in_conn, in_monitor, out_monitor,
                          v_mon)

    if self.inhibit: network.add(self.inhibit)

    network.reinit()
    network.run(trial.duration()*ms)
    spikes_in  = self.monitor_to_signal(trial, signal.dims(), in_monitor)
    spikes_out = self.monitor_to_signal(trial, self.p['neurons']['N'], out_monitor)

    return Result(self.p, signal, spikes_in, spikes_out)


  def monitor_to_signal(self, trial, N, monitor):
    list = []
    for i in xrange(N):
      list.append(monitor[i])
    return SparseBinarySignal(trial, list)


class Result:
  ''' Object variables:
        - params: Dictionary of parameters used to initialize the simulation
        - input:  Input signal
        - output: Tuple of output signals from neuron populations (usually
            input Poisson population and output Izhikevich population)
        - trial:  Clock information '''

  def __init__(self, params, signal, input, output):
    self.params = params
    self.signal = signal
    self.input  = input
    self.output = output 
  
  def write_to_file(self,filename):
    cPickle.Pickler(file(filename,'w'),2).dump(self)
  
