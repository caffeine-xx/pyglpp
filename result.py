import cPickle
from NeuroTools import signals
from matplotlib import pylab

class Result:
  ''' Object variables:
        - params: Dictionary of parameters used to initialize the simulation
        - input:  Input signal
        - output: Tuple of output signals from neuron populations (usually
            input Poisson population and output Izhikevich population)
        - trial:  Clock information '''

  def __init__(self, params, signal, input, output, monitors={}):
    self.params = params
    self.signal = signal
    self.input  = input
    self.output = output 
    self.monitors = monitors

  def write_to_file(self,filename):
    cPickle.Pickler(file(filename,'w'),2).dump(self)
  
class NeuroToolsResult(Result):
  ''' Converts a normal result into NeuroTools signals '''
  def __init__(self, res):
    self.signal = res.signal.to_analog()
    self.input  = signals.SpikeList(self.de_unit(res.input))
    self.output = signals.SpikeList(self.de_unit(res.output))
    self.monitors = {}
    for k in self.monitors:
      self.monitors[k] = signals.AnalogSignalList(res.monitors[k], res.signal.trial.dt)

  def de_unit(self, list):
    return [(id, time / ms) for (id, time) in list]

  def graph(self):
    for k in self.monitors:
      self.monitors[k].plot(ylabel=k)
    self.input.raster_plot('input spikes')
    self.output.raster_plot('output spikes')
    self.signals.plot(ylabel='signal')
    pylab.show()

def load_result(filename):
    return cPickle.Unpickler(file(filename, 'r')).load()

