import numpy as np
import math as m
import scipy.signal as sig

# a single basis row
def cos_basis(a, c):
  phi = lambda n,j: j * m.pi / (2)
  dis = lambda t: a * m.log(t + c)
  bas = lambda n,j,t: (dis(t) > (phi(n,j) - m.pi) and dis(t) < (phi(n,j) + m.pi)) * ((1.0/2.0)*(1 + m.cos(dis(t) - phi(n,j))))
  return np.vectorize(bas)

def straight_basis(a):
  bas = lambda n, j, t: a
  return np.vectorize(bas)

def run_bases(bases, data):
  """Correlates a dataset with a set of bases.
   Takes a 2D array to a 3D array, """
  rows,cols = data.shape
  num,size = bases.shape
  result = np.zeros([rows,cols,num])
  for i in range(0,num):
    result[:,:,i] = run_filter(data,bases[i])
  return result

# runs a filter row-wise on some data
def run_filter(data, filter):
  rows,cols = data.shape
  corr = lambda i: sig.correlate(data[i],filter,mode='same')
  return np.array([corr(i) for i in range(0,rows)])
