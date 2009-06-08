import numpy as np
import math

def cos_filt(size):
  base = map(lambda i: i * math.pi / (2*size), range(0,size))
  return np.array(map(lambda i: math.cos(i), base))

def rand_filt(size):
  return np.array([rd.random() for i in range(size)])
