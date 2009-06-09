import numpy as np
import math as m
import scipy.signal as sig

# a single basis row
def cos_basis(a, c):
  phi = lambda n,j: j * m.pi / (2 * n)
  dis = lambda t: a * m.log(t + c)
  bas = lambda n,j,t: (dis(t) > (phi(n,j) - m.pi) and dis(t) < (phi(n,j) + m.pi)) * ((1.0/2.0)*(1 + m.cos(dis(t) - phi(n,j))))
  return np.vectorize(bas)

def straight_basis(a):
  bas = lambda n, j, t: a
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


def run_bases(bases, data):
"""Correlates a dataset with a set of bases.
   Takes a 2D array to a 3D array, where the 
   dimensions are [basis, rows, cols]"""
  rows,cols = data.shape
  num,size = bases.shape
  result = np.zeros([num,rows,cols])
  corr = lambda i,j: sig.correlate(data[j],bases[i])
  for i in range(0,num):
    for j in range(0,rows):
      result[i,j,:] = corr(i,j)
  return result

# runs a filter on some data
def run_filter(data, filter):
  rows,cols = data.shape
  corr = lambda i: sig.correlate(data[i],filter[i])
  return np.array([corr(i) for i in range(0,rows)])

# sums together the values at particular times
# time is the 2nd dimension
# sums up the data in those cols
def selective_sum(filtered, times,axis=0):
  return np.sum(filtered[:,tuple(times)],axis=axis)
