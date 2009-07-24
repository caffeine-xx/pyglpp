import result
import numpy as np
from signals import *
from inference import *
from scipy import io
from utils import print_timing
from result import load_result
from information import *

def run_analysis(prefix,  model=False):
  ''' Analyzes a particular trial and saves the result in
      filters in prefix.mat (MATLAB-compatible) '''
  if not model: model = standard_model()
  experiment = load_result(prefix+".pickle")
  inferred   = analyze_experiment(model, experiment)
#  inferred.update(information_analysis(inferred))
  io.savemat(prefix+".mat",inferred)
  return inferred

@print_timing
def analyze_experiment(model,experiment):
  ''' Performs ML inference on the given model,
  and returns the resulting parameters '''
  
  trial = experiment.signal.trial
  input = experiment.signal
  output = SparseBinarySignal(trial,experiment.output)
  model.set_data(trial,input,output)
  initial   = model.random_args()
  estimator = MLEstimator(model)
  maximized = estimator.maximize(*initial)
  intensity = model.logI(*maximized)
  result    = dict(zip(('K','H','Mu','T','logI','X','Y','Yr','Xb','Yb'),
              maximized + (trial.range(),intensity,input.signal,output(),
                           array(experiment.output),model.stim_basis,
                           model.spike_basis)))
  return result

@print_timing
def information_analysis(dat):
  ''' Calculates mutual info and transfer entropy between:
      - Each pair of neurons (in each direction)
      - The input signal and each neuron
      The neuron signal analyzed is the log-Poisson intensity'''
  bins = 5
  lag = 2
  #MI = mutual_information(dat['logI'],dat['logI'],bins=bins)
  TE_X = np.array([[transfer_entropy(x,y,lag=lag,bins=bins) 
                    for y in dat['X']] for x in dat['logI']])
  TE_I = np.array([[transfer_entropy(x,y,lag=lag,bins=bins)
                    for y in dat['logI']] for x in dat['logI']])
  return { 'TE_X':TE_X, 'TE_I':TE_I}

def standard_model():
  model      = SimpleModel()
  return model

if(__name__=="__main__"):
  import sys

  try:
    prefix = sys.argv[1]
  except:
    prefix = "neu2"

  prefs = ["results/"+prefix] 
  id = 0
  end_id = 0

  if(len(sys.argv)>2):
    id = int(sys.argv[2])
    end_id = id+1
    prefs = []
  if len(sys.argv)>3:
    end_id = int(sys.argv[3])+1

  if not id==0:
    prefs = ["results/%s_%i" % (prefix, i) for i in range(id, end_id)]

  for p in prefs:
    print "=== Running analysis: %s" % p
    run_analysis(p)
