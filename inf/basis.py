import numpy as np
import math as m
import scipy.signal as sig

# a single basis row
def cos_basis(a, c):
  phi = lambda n,j: j * m.pi / (2 * n)
  dis = lambda t: a * m.log(t + c)
  bas = lambda n,j,t: (dis(t) > (phi(n,j) - m.pi) and dis(t) < (phi(n,j) + m.pi)) * ((1.0/2.0)*(1 + m.cos(dis(t) - phi(n,j))))
  return np.vectorize(bas)

# for single column/row of coeffs only
def combine_bases(coeffs, tau, basis):
  filter = np.zeros(tau.shape)
  for i,v in enumerate(coeffs):
     filter = filter + v * basis(coeffs.size, i, tau)
  return filter

# assembles a multi-dimensional filter
def filter_builder(arr, tau, basis):
  rows, cols = arr.shape
  filter = np.zeros([rows,tau.size])
  for i in range(0,rows):
    filter[i,:] = combine_bases(arr[i,:], tau, basis)
  return filter

# runs a filter on some data
def run_filter(data, filter):
  rows,cols = data.shape
  corr = lambda i: sig.correlate(data[i],filter[i])
  return np.array([corr(i) for i in range(0,rows)])

# sums together the values at particular times
# time is the 2nd dimension
# sums up the data in those rows
def selective_sum(filtered, times):
  return sum(filtered[:,tuple(times)])
