from numpy import *
from inference import *

import runner as r
reload(r)

def test_analyze():
  basis = atleast_2d([0.5, 0.5])
  model = MultiNeuron(basis, basis)
  prefix = "results/poisson_stim_rand_inh"
  result = r.analyze_experiment(prefix,model)
  return result


if (__name__ == "__main__"):
  print test_runner()

