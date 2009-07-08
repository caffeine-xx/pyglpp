from numpy import *
from brian import *
from brian.library.IF import *
from signals import *
import cPickle

class Simulator:
  ''' Wrapper for the Brian simulator '''
  
  eqs = '''
        dv/dt  = ((0.04/mV)*v**2 + 5.0*v + 140.0*mV - w)/ms + I/C_m : volt
        dw/dt  = a*(b*v - w)/ms : volt
        I      = ge*(Ee-v) - gi*(Ei-v) : amp
        dge/dt = -ge / te : siemens
        dgi/dt = -gi / ti : siemens'''
   
  def __init__(self, model  ={'a':0.02,        'b':0.2, 
                              'Ee':0*mV,       'Ei':-80.0*mV, 
                              'te':10*ms,      'ti':10*ms,
                              'C_m':2.0*pF                 },
                     init   ={'gei':10*nS,     'gii':20*nS },
                     neurons={'N':20,          'Ni': 7,
                              'threshold':     'v > 30.0*mV',
                              'reset':         'v = -65.0*mV;'+
                                               'w += 8.0*mV'},
                     connect={'weight':0.1*nS, 'sparseness':0.1,
                              'state':'ge'                 },
                     inhibit={'weight':0.1*nS, 'sparseness':0.2,
                              'state':'gi'}):
    (N, Ni) = (neurons['N'], neurons['Ni'])
    self.params  = {'model':model,     'neurons':neurons, 
                    'connect':connect, 'inhibit':inhibit, 'init':init}
    self.model   = Equations(self.eqs, **model)
    self.neurons = NeuronGroup(model=self.model,**neurons)
    self.connect = Connection(self.neurons[Ni:], self.neurons, **connect)
    self.inhibit = Connection(self.neurons[:Ni], self.neurons, **inhibit)
    
  def run(self, signal=GaussianNoiseGenerator(40.0, 10.0, 10).generate(Trial(0.0,100.0,0.1)), 
                connect_in={'state':'ge','weight':10*nS, 'delay':1*ms}):
    ''' Runs a trial of the simulation using the input signal provided '''
    trial       = signal.trial
    input       = PoissonGroup(signal.dims(), rates=lambda t: signal[t])
    in_conn     = Connection(input, self.neurons, **connect_in)
    in_monitor  = SpikeMonitor(input, record=True)
    out_monitor = SpikeMonitor(self.neurons, record=True)
    v_mon       = StateMonitor(self.neurons, 'v', record=True)
    network     = Network(self.neurons, self.connect, self.inhibit,
                          input, in_conn, in_monitor, out_monitor,
                          v_mon)
    network.reinit()
    network.run(trial.duration()*ms)
    
    self.params['connect_in'] = connect_in
    
    spikes_in  = self.monitor_to_signal(trial, signal.dims(), in_monitor)
    spikes_out = self.monitor_to_signal(trial, self.params['neurons']['N'], out_monitor)
    result = Result(self.params, signal, spikes_in, spikes_out)
    return result
  
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
  
