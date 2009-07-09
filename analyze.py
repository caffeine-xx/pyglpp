import result
from signals import *
from inference import *
from scipy import io

def run_analysis(prefix,  model=False):
  ''' Analyzes a particular trial and saves the result in
      filters in prefix.mat (MATLAB-compatible) '''
  if not model: model = standard_model()
  experiment = result.load_result(prefix+".pickle")
  inferred   = analyze_experiment(model, experiment)
  save_parameters(prefix+".mat", inferred)
  return inferred

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
  
  return maximized

def standard_model():
  length     = Trial(0,10.0,0.1)
  stim_bas   = SineBasisGenerator(2.5,4).generate(length)
  spike_bas  = SineBasisGenerator(7, 10).generate(length)
  model      = SimpleModel(stim_bas, spike_bas)
  return model

def load_parameters(filename):
  ''' Loads a set of poisson parmaeters from a file '''
  p = io.loadmat(filename)
  return param_from_dict(p)

def save_parameters(filename, parameters,model=None):
  ''' Saves a set of poisson parameters to a file '''
  params = param_to_dict(parameters)
  if model != None:
    params['Xb'] = model.stim_basis
    params['Yb'] = model.spike_basis
  io.savemat(filename,params)

def param_to_dict(parameters):
  return dict(zip(('K','H','Mu'),parameters))

def param_from_dict(p):
  return (p['K'],p['H'],p['Mu'])

if(__name__=="__main__"):
  import sys

  prefix = sys.argv[1]

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
    print run_analysis(p)
