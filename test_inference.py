import numpy as np
from scipy import *
import math as m

from inference import *
from stimulus import *
from signals import *

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

  bx =  np.array(
        [[[ 0.0,  0.0],
          [ 1.0,  2.0],
          [ 1.0,  1.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 2.0,  4.0],
          [ 2.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]]])
  by = np.array(
        [[[ 0.0,  0.0],
          [ 1.0,  2.0],
          [ 1.0,  1.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 1.0,  2.0],
          [ 1.0,  1.0],
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
  vI = np.array([[0,6,4,0, 9,6,0,0],
                 [0,12,8,0, 18,12,0,0]])
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

  return mn

def test_logL_hess():
  import inference as inf
  reload(inf)

  b = np.array([
    [1,1],
    [2,1]])
  mn = inf.MultiNeuron(b,b)

  t = 1
  d = 8
  x = np.array([
    [0,1,0,0, 0,0,0,0],
    [0,0,0,0, 2,0,0,0]])
  s = [[1],[4]]
  mn.set_data(t,d,x,s)

  # make sure it doesn't die
  (K,H,M) = mn.random_args()
  (K1,H1,M1) = mn.random_args()
  (dK,dH,dM) = mn.logL_grad(K,H,M)
  (PK,PH,PM) = mn.logL_hess_p(K,H,M,K1,H1,M1)
  shapechk = lambda (a,b): a.shape == b.shape

  assert map(shapechk,zip([K,H,M],[K1,H1,M1]))
  assert map(shapechk,zip([K,H,M],[PK,PH,PM]))
  assert map(shapechk,zip([dK,dH,dM],[PK,PH,PM]))

  # make sure wrapper works
  theta, args = mn.pack(K,H,M)
  p, args = mn.pack(K1,H1,M1)
  mles = inf.MLEstimator(mn)
  mles.logL_hess_p(theta,p,*args)
  
  # make sure it still gives back decent maximizations
  a0 = mn.random_args()
  p1 = mles.maximize(*a0)
  p2 = mles.maximize_cg(*a0)
  t1 = mn.pack(*p1)[0]
  t2 = mn.pack(*p2)[0]
  print t1, t2, np.corrcoef(t1,t2)
  assert abs(mn.logL(*p1) - mn.logL(*p2)) < 0.1
  assert np.sqrt(np.sum(t1 - t2)**2) < 0.1

def test_simplemodel():

  b = np.array([
    [1,1],
    [2,1]])
  b = Signal(Trial(0,2,1),b)

  mn = SimpleModel(b,b)

  t = Trial(0,8,1)
  x = np.array([
    [0,1,0,0, 0,0,0,0],
    [0,0,0,0, 2,0,0,0]])
  s = [(0,1),(1,4)]

  x = Signal(t,x)
  s = SparseBinarySignal(t,s)
  
  mn.set_data(t,x,s)

  b = b()
  s = s.sparse_bins()

  bx =  np.array(
        [[[ 0.0,  0.0],
          [ 1.0,  2.0],
          [ 1.0,  1.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 2.0,  4.0],
          [ 2.0,  2.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]]])
  by = np.array(
        [[[ 0.0,  0.0],
          [ 1.0,  2.0],
          [ 1.0,  1.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0]],
         [[ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 0.0,  0.0],
          [ 1.0,  2.0],
          [ 1.0,  1.0],
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
  vI = np.array([[0,6,4,0, 9,6,0,0],
                 [0,12,8,0, 18,12,0,0]])
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
  test_logL_hess()
  test_multineuron()
  test_simplemodel()
  test_mlestimator()
  test_pack_unpack()
  
