import numpy as np
import math

def logI(t, theta, data):
  return np.reshape(theta * data(t).T,[1])

def logL(theta, data, delta, time, size, sp_times):
  intensities = [logI(t, theta, data) for t in range(size,time)]
  term1 = sum([intensities[t-size] for t in sp_times])
  term2 = sum([delta*math.exp(intensities[t-size]) for t in range(size,time)])
  return np.array(term1-term2)

def logL_grad(theta, data, delta, time, size, sp_times):
  term1 = sum([data(t) for t in sp_times])
  term2 = sum([delta * math.exp(logI(t,theta,data)) * data(t) for t in range(size,time)])
  result = np.array(term1-term2)
  return np.reshape(result, [result.size])

def logL_hess(theta, data, delta, time, size, sp_times):
  datm = lambda t: np.matrix(data(t))
  dsqu = lambda t: datm(t).T * datm(t)
  result = -1 * delta * sum([dsqu(t)*math.exp(logI(t,theta,data)) for t in range(size,time)])
  return np.asarray(np.reshape(result,[theta.size,theta.size]))

def logL_hess_p(theta, p, *args):
  hessian = np.asmatrix(logL_hess(theta,*args))
  p = np.asmatrix(p)
  return np.asarray(p * hessian )
