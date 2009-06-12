
from numpy.random import poisson
import random as rd
import math
import numpy as np
import basis as b


class Stimulator:
  """ Stimulator class.  Supports both incremental and 
  range-based stimulus generation.  Returns 
  Nx by T sized stimulus arrays (T=1 for incremental) """

  def __init__(self, delta, Nx):
    self.d = delta
    self.T = 0
    self.Nx = Nx
    self.X = np.zeros([Nx, 100])

  def next(self):
    raise Exception("Not implemented.")
  
  def range(self,t0,t1):
    raise Exception("Not implemented.")


class CosineStim(Stimulator):
  """ It's all about the cosine.  Periods and phases can be specified
  but are usually zero - multiples of 2pi."""

  def __init__(self, delta, Nx, periods=[1], phases=[0]):
    assert len(periods)==len(phases)
    if (len(periods)==1):
      periods = Nx * periods
      phases = Nx * phases
    assert len(periods)==Nx
    Stimulator.__init__(self,delta, Nx)
    self.pe = periods
    self.ph = phases

  def cosine(self,period=(math.pi*2),phase=0):
    return np.vectorize(lambda t: math.cos((phase+t)*math.pi*2*period)**2)

  def simulate(self, time):
    return np.asarray([self.cosine(pe,ph)(time) for pe,ph in zip(self.pe,self.ph)])
  
  def range(self, t0, t1):
    tau = np.arange(t0, t1, self.d)
    return self.simulate(tau)

  def next(self):
    self.T += self.d
    return self.simulate(self.T)

class RandomStim(Stimulator):
  """ Uniform stimulator """

  def range(self, t0, t1):
    return np.random.rand(self.Nx,(t1-t0)/self.d)

  def next(self):
    return np.random.rand(self.Nx,1)
