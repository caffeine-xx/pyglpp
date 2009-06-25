from numpy import *
from inference import *

import analyze as r
reload(r)

def test_analyze():
  basis = cos_basis(n=4)
  model = MultiNeuron(basis,basis)
  prefix = "results/single_izhikevich_trial"
  experiment = r.load_brian_experiment(prefix)
  result = r.analyze_experiment(model,experiment)
  r.save_parameters("results/params.mat",result)
  return r.param_to_dict(result)

if (__name__ == "__main__"):
  print test_analyze()

