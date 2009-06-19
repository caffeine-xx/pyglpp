import psyco
import timeit

from numpy import *
from runner import analyze_experiment
from inference import MultiNeuron

psyco.full()
def profile_me():
  model = MultiNeuron(atleast_2d([0.5,0.5]), atleast_2d([0.5,0.5]))
  prefix = "results/poisson_stim_rand_inh"
  results = analyze_experiment(prefix,model)
  return results

if(__name__=="__main__"):
  print profile_me()
