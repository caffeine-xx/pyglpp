import sys
from trials import *
from analyze import *
import cPickle

def run_random_network_trial(prefix,id):
  ''' Runs a trial (a simulation) from trials.py
      Generates parameters from prefix_params, runs prefix_trial,
      and saves the relevant info in results/prefix_id_P.dat '''
  params = random_network_params(prefix,id)
  random_network_trial(**params)
  cPickle.dump(params, file("results/%s_%i_P.dat" % (prefix, id), 'w'), 2)

def run_analysis(prefix, id):
  ''' Analyzes a particular trial and saves the result in
      filters in prefix_id_R.mat (MATLAB-compatible)
      Bases are currently hardcoded i, need to find neater
      solution. '''
  params = cPickle.load(file("results/%s_%i_P.dat" % (prefix, id), 'r'))
  experiment = load_brian_experiment(params['prefix'])
  model      = standard_model()
  result     = analyze_experiment(model, experiment)
  save_parameters("results/%s_%i_R.mat" % (prefix,id), result)

if(__name__ == "__main__"):
    for i in xrange(int(sys.argv[1]),int(sys.argv[2])):
      print "====\tRunning network: \t%i"% i
      run_random_network_trial("random_network",i)
      print "====\tAnalyzing network: \t%i" % i
      run_analysis("random_network",i)
