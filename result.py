import config
import cPickle
import signals as sig
import numpy as np
from matplotlib import pyplot
from scipy import io

class ResultPlotter:
  def __init__(self, result):
    self.result = result

  def plot(self):
    pyplot.figure()
    time = self.result.trial.range()
    [pyplot.plot(time, k) for k in self.result.stimulus()]
    pyplot.figure()
    pyplot.subplot(211)
    self.raster(self.result.input, label='input')
    pyplot.subplot(212)
    self.raster(self.result.output, label='output')
    pyplot.figure()
    for n,k in enumerate(self.result.monitors):
        pyplot.subplot(len(self.result.monitors), 1, n+1)
        [pyplot.plot(x) for x in self.result.monitors[k]]
    pyplot.show()

  def raster(self, train, **args):
    if len(train)==0: return None
    ids = np.zeros(len(train))
    tim = np.zeros(len(train))
    for k,(i,t) in enumerate(train):
      ids[k] = i
      tim[k] = t
    pyplot.plot(tim, ids, color='white', marker='o', 
                  markerfacecolor='blue',markersize=3,  **args)

class Result:

  def __init__(self, params, stimulus, input, output, monitors={}):
    self.params = params
    self.trial  = stimulus.trial
    self.input  = self.de_unit(input)
    self.output = self.de_unit(output)
    self.monitors = monitors
    self.stimulus = stimulus
    self.input_trains = sig.SparseBinarySignal(self.trial,self.input)
    self.spike_trains = sig.SparseBinarySignal(self.trial,self.output)

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
      self.input_trains = self.input_trains
      self.spike_trains = self.spike_trains
    except:
      self.input_trains = sig.SparseBinarySignal(self.trial,self.input)
      self.spike_trains = sig.SparseBinarySignal(self.trial,self.output)
    exp = {
      'input_weight':self.params['inputs']['weight'],
      'connect_weight':self.params['connect']['weight'],
      'timeline':self.trial.range(),'stimulus':self.stimulus.signal,
      'input_trains':self.input_trains.fill(), 'input_raster':array(self.input),
      'spike_trains':self.spike_trains.fill(), 'spike_raster': array(self.output),
    }
    exp.update(getattr(self,imported,{}))
    return exp

def load_result(filename):
  return cPickle.Unpickler(file(filename+".pickle", 'r')).load()

def save_result(filename,result):
  cPickle.Pickler(file(filename+".pickle",'w'),2).dump(result)

def export_result(filename,result):
  io.savemat(filename,result.export_dict())

if(__name__=="__main__"):
  import sys
  prefix = sys.argv[1]
  res = ResultPlotter(load_result(config.results_dir + prefix))
  res.plot()

