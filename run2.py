import cProfile
from inference import *
from runner import *

def profile_me():
  model = MultiNeuron(atleast_2d([0.5,0.5]), atleast_2d([0.5,0.5]))
  prefix = "results/poisson_stim_rand_inh"
  results = analyze_experiment(prefix,model)
  return results

if(__name__ == "__main__"):
  cProfile.run("profile_me()", "exprof")

