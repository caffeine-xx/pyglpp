import cPickle
import signals as sig
import NeuroTools.signals as ntsig
from brian import *
from matplotlib import pyplot
from scipy import io

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
    pyplot.plot(tim, ids, color='white', marker='o', 
                  markerfacecolor='blue',markersize=3,  **args)

  def de_unit(self, list):
    return [(id, float(time)) for (id, time) in list]

  def import_dict(self, data):
    if not getattr(self, 'imported', False):
      self.imported = dict()
    self.imported.update(data)
    for k in data.keys():
      setattr(self, k, data[k])

  def export_dict(self):
    try: 
      input_trains = self.input_trains
      spike_trains = self.spike_trains
    except:
      input_trains = sig.SparseBinarySignal(self.signal.trial,self.input)
      spike_trains = sig.SparseBinarySignal(self.signal.trial,self.output)

    exp = {
      'input_weight':self.params['inputs']['weight'],
      'connect_weight':self.params['connect']['weight'],
      'timeline':self.signal.trial.range(),'stimulus':self.signal.signal,
      'input_trains':input_trains, 'input_raster':array(self.input),
      'spike_trains':spike_trains, 'spike_raster': array(self.output),
    }

    exp.update(getattr(self,imported,{}))

    return exp

def load_result(filename):
  return cPickle.Unpickler(file(filename+".pickle", 'r')).load()

def save_result(prefix,result):
  cPickle.Pickler(file(filename+".pickle",'w'),2).dump(result)

def export_result(prefix,result):
  io.savemat(prefix,result.export_dict())

if(__name__=="__main__"):
  import sys
  prefix = sys.argv[1]
  res = load_result("results/%s" % prefix)
  res.plot()

