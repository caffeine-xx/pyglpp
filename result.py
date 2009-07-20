import cPickle
from matplotlib import pyplot
from brian import *
from NeuroTools import signals

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
    self.input  = self.de_unit(input)
    self.output = self.de_unit(output)
    self.monitors = monitors
  
  def write_to_file(self,filename):
    cPickle.Pickler(file(filename,'w'),2).dump(self)

  def plot(self):
    pyplot.figure()
    time = self.signal.trial.range()
    [pyplot.plot(time, k) for k in self.signal()]
    pyplot.figure()
    pyplot.subplot(211)
    self.raster(self.input, label='input')
    pyplot.subplot(212)
    self.raster(self.output, label='output')
    pyplot.figure()
    for n,k in enumerate(self.monitors):
        pyplot.subplot(len(self.monitors), 1, n+1)
        [pyplot.plot(x) for x in self.monitors[k]]
    pyplot.show()

  def raster(self, train, **args):
    if len(train)==0: return None
    ids = zeros(len(train))
    tim = zeros(len(train))
    for k,(i,t) in enumerate(train):
      ids[k] = i
      tim[k] = t
    pyplot.plot(tim, ids, color='white', marker='o', markerfacecolor='blue',
      markersize=3,  **args)

  def de_unit(self, list):
    return [(id, float(time)) for (id, time) in list]

class NeuroToolsResult(Result):
  ''' Converts a normal result into NeuroTools signals '''
  def __init__(self, res):
    self.signal = res.signal.to_analog()
    ids = range(res.params['neurons']['N'])
    self.input  = signals.SpikeList(self.de_unit(res.input), range(res.signal.dims()))
    print self.input.spikes
    self.output = signals.SpikeList(self.de_unit(res.output), ids)
    print self.output.spikes
    self.monitors = {}
    for k in res.monitors:
      self.monitors[k] = signals.AnalogSignalList(res.monitors[k]/ms, ids, **res.signal.trial.to_hash())

  def graph(self):
    for k in self.monitors:
      for j in self.monitors[k]:
        j.plot(ylabel=k)
    self.input.raster_plot()
    self.output.raster_plot()
    self.signal.plot(ylabel='signal')
    pylab.show()

def load_result(filename):
    return cPickle.Unpickler(file(filename, 'r')).load()

