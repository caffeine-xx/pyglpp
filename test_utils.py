import numpy as np
import math as m
import matplotlib.pyplot as plt
from inference import BasisUtils
from scipy import *

b = BasisUtils()

def test_cos_basis():
  tau = np.arange(0.1,30.0,0.10)
  bas = b.cos_basis(7,1.0)
  bas2 = b.cos_basis()
  plt.hold(True)
  for j in range(0,10):
    plt.plot(bas(10,j,tau))
    plt.show()
    assert bas2(10,j,tau).all()==bas(10,j,tau).all()

def test_run_filter():
  # simple multiplication
  filter = np.array([2,0])
  data = np.ones([2,10])
  newdata = b.run_filter(filter,data)
  assert  (2*data).all() == newdata.all()

def test_run_bases():
  filters = np.array([[1,0],[2,0],[3,0]])
  data = np.ones([2,10])
  newdata = b.run_bases(filters,data)
  print newdata
  valdata = np.array([data, 2*data, 3*data])
  assert newdata.all() == valdata.all()

