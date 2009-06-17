import numpy as np
import pyNN.nest as pn

class Simulator():
  def __init__(self,dt=1.0):
    """ Initialize a simple sizexsize population of IF neurons,
    and interconnect each pair with probability p. 
    Connect neurons randomly to a single Poisson source with probability p."""
    pn.setup(timestep=dt)
    self.RNG  = pn.NumpyRNG() 

 def _fixed_connector(self, p):
    return pn.FixedProbabilityConnector(p,allow_self_connections=True,weights=1.0)
 
 def _if_exp_population(self, size, *params):
    return pn.Population(size, pn.IF_cond_exp, *params)

 def _dist_sq_connector(self,m,*args):
    fn = lambda d: 1 - (d/m)**2
    return pn.DistanceDependentProbabilityConnector(fn, *args)

 def simulation(self):
    raise Exception("Not implemented. Define your own simulation here!")

