from numpy import *
from brian import *
from brian.library.IF import *
from signals import *

class Simulator:
  ''' Wrapper for the Brian simulator '''
  def __init__(self, model={'a':0.2/ms, 'b':0.2/ms}, 
                     neurons={'N':100, 'Vr':-75.0*mV, 'b':0.2/ms},
                     connect={'weight':0.2*nS, 'sparseness':0.5,
                              'delay':(0*ms,5*ms), 'max_delay':5*ms}):
    self.model   = Izhikevich(**model)
    self.neurons = NeuronGroup(model=self.model, **neurons)
    self.connect = Connection(self.neurons, self.neurons, **connect)
    self.params  = {'model':model, 'neurons':neurons, 'connect':connect }
  
  def run(self, input_signal, input_conn):
    ''' Runs a trial of the simulation using the 
        input signal provided '''
    self.params['stimulus'] = input_conn
    trial = input_signal.trial
  
    self.signal      = input_signal
    self.clock       = Clock(dt=trial.dt*ms)
    self.input       = PoissonGroup(input_signal.dims(), rates=lambda t: input_signal[t],
                        clock=self.clock)
    self.stimulus    = Connection(input, neurons, **input_conn)
    self.in_monitor  = SpikeMonitor(self.input)
    self.out_monitor = SpikeMonitor(self.output)
  
    self.neurons.clock = self.clock
    self.__callback__(input_signal, input_conn)
    run(trial.duration()*ms)
    
    spikes_in  = self.monitor_to_signal(self.trial, input_signal.dims(), self.in_monitor)
    spikes_out = self.monitor_to_signal(self.trial, self.params['neurons']['N'], self.out_monitor)
    
    result = Result(self.params, input_signal, (spikes_in, spikes_out), trial)
    return result
  
  def monitor_to_signal(self, trial, N, monitor):
    list = []
    for i in xrange(N):
      list.append(monitor[i])
    return SparseBinarySignal(trial, list)
  
  def __callback__(self):
    ''' Allows for customization of initialization - usually
        weights and other complicated connectivity will be done here.'''
    return None

class Result:
  ''' Object variables:
        - params: Dictionary of parameters used to initialize the simulation
        - input:  Input signal
        - output: Tuple of output signals from neuron populations (usually
            input Poisson population and output Izhikevich population)
        - trial:  Clock information '''
  def __init__(self, params, input, output, trial):
    self.params = params
    self.input  = input
    self.output = output 
    self.trial  = trial
  
  def write_to_file(self,filename):
    # TODO: Implement
    raise Exception("Unimplemented method")

def run_simulation(input_signal, simulator, times=1):
  for t in trials:
    simulator.run(input_signal)
  simulator.write('output.dat')



def single_izhikevich_trial(a=0.2/ms,b=0.2/ms,rate=40.0*Hz,deviation=20.0*Hz,time=10000*ms,prefix='results/single_izhikevich_trial'):
  ''' Runs a trial of an izhikevich neuron with a single
      randomly rate-varying Poisson input, centered at 40Hz.'''
  # Izhikevich model with excitatory synapse
  model  = Izhikevich(a,b)
  reset  = AdaptiveReset(Vr=-75*mV, b=0.2/ms)
  neuron = NeuronGroup(1,model=model, threshold=-30.0*mV, reset=reset)
    
  # Poisson stimulus
  rates           = lambda t: max(0,normal(rate, deviation))
  stimulus        = PoissonGroup(1,rates=rates)
  connection      = Connection(stimulus, neuron)
  connection[0,0] = 40*mvolt
  
  # Spike recording
  in_file  = "%s_S.ras"  % (prefix)
  out_file = "%s_N.ras" % (prefix)
  
  in_monitor  = NeuroToolsSpikeMonitor(stimulus,in_file,record=True)
  out_monitor = NeuroToolsSpikeMonitor(neuron,out_file,record=True)
  
  neuron.clock.reinit()
  run(time)
