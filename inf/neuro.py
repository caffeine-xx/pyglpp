import numpy as np
from memoize import *
from trains import *
from slicers import *
from likelihood import *

def maxL(opt, delta, size, spikes, stim):
  theta = filt_slice(0, np.zeros(size), np.zeros(size))
  times = filter(lambda t: t>size, spike_times(spikes))
  data = Memoize(data_slicer(size, stim, spikes))
  args = (data, delta, spikes.size, size, times)
  f = lambda theta: -1 * logL(theta, *args)
  fp = lambda theta: -1 * logL_grad(theta, *args)
  fh = lambda theta, p: -1 * logL_hess_p(theta, p, *args)
  return opt(f=f, x0=theta, fprime=fp, fhess_p=fh)
