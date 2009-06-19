import numpy as np
from scipy import *
import math as m

from inference import *
from stimulus import *

# --------------
# bases tests
# --------------

def test_cos_basis():
  tau = np.arange(0.1,30.0,0.10)
  bas = cos_basis(7,1.0)
  bas2 = cos_basis()
  for j in range(0,10):
    assert bas2(10,j,tau).all()==bas(10,j,tau).all()

def test_run_filter():
  # simple multiplication
  filter = np.array([2,0])
  data = np.ones([2,10])
  newdata = run_filter(filter,data)
  assert  (2*data).all() == newdata.all()

def test_run_bases():
  filters = np.array([[1,0],[2,0],[3,0]])
  data = np.ones([2,10])
  newdata = run_bases(filters,data)
  print newdata
  valdata = np.array([data, 2*data, 3*data])
  assert newdata.all() == valdata.all()

# --------------
# multineuron tests
# --------------

def test_pack_unpack():
  neurons  = 2
  stimuli  = 2
  bas = atleast_2d([0.5, 0.5])
  K = np.random.random([neurons,stimuli,4])
  H = np.random.random([neurons,neurons,4])
  Mu = np.random.random([neurons])
  mn = MultiNeuron(bas, bas)
  th,ar = mn.pack(K,H,Mu)
  oK,oH,oM = mn.unpack(th,ar)
  assert (K==oK).all() and (H==oH).all() and (Mu==oM).all()

def test_multineuron():
  b = np.array([
    [1,1],
    [2,1]])
  mn = MultiNeuron(b,b)

  t = 1
  d = 8
  x = np.array([
    [0,1,0,0, 0,0,0,0],
    [0,0,0,0, 2,0,0,0]])
  s = [[1],[4]]
  mn.set_data(t,d,x,s)

  #  y = np.array([
  #    [0,1,0,0, 0,0,0,0],
  #    [0,0,0,0, 1,0,0,0]])

  bx =  np.array([[[ 1.0,  1.0],
          [ 1.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 2.0,  2.0],
          [ 2.0,  4.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]]])
  by = np.array([[[ 1.0,  1.0],
          [ 1.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 1.0,  1.0],
          [ 1.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]]])

  assert (bx==mn.base_stims).all()
  assert (by==mn.base_spikes).all()

  # check intensities
  K = np.ones((2, 2, 2))
  K[1,:,:] = K[1,:,:] * 2
  H = np.ones((2, 2, 2))
  H[1,:,:] = H[1,:,:] * 2
  M = np.zeros(2)
  I = mn.logI(K,H,M)
  vI = np.array([[4,6,0,6, 9,0,0,0],
                 [8,12,0,12, 18,0,0,0]])

  assert (I==vI).all()

  # check likelihoods
  l0 = vI[0,1] - np.sum(np.exp(vI[0,:]))
  l1 = vI[1,4] - np.sum(np.exp(vI[1,:]))
  L  = l0 + l1
  cL = mn.logL(K,H,M)

  assert (L - mn.logL(K,H,M)) < 0.001

  # check gradients
  eI = np.exp(I)
  dK, dH, dM = mn.logL_grad(K,H,M)
  for i in range(0,2):
    for j in range(0,2):
      for l in range(0,2):
        sp = s[i][0]
        g = bx[j,sp,l] - np.sum(bx[j,:,l] * eI[i,:])
        assert abs(g-dK[i,j,l]) < 0.01
        g = by[j,sp,l] - np.sum(by[j,:,l] * eI[i,:])
        assert abs(g-dH[i,j,l]) < 0.01
        


# ------------
# maximum likelihood tests
# ------------

class LLStub(LikelihoodModel):
  def logL(self,x,n,c):
    return -1*(x**n)+c

  def logL_grad(self,x,n,c):
    return -1*n*(x**(n-1)),0,0

  def pack(self,x,n,c):
    theta=x
    args=(n,c)
    return theta,args

  def unpack(self,theta,args):
    x=theta
    (n,c)=args
    return x,n,c

def test_mlestimator():

  lls = LLStub()
  c = 5
  n = 2
  x0 = 10

  est = MLEstimator(lls)
  (x,n,c) = est.maximize(x0,n,c)
  
  assert x<0.01

if (__name__ == "__main__"):
  test_multineuron()
  test_mlestimator()
  test_pack_unpack()

