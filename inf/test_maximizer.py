
import numpy as np
import math as m

from scipy import *

import likelihood_model as lm
import mlestimator as mle

reload(lm)
reload(mle)

class LLStub(lm.LikelihoodModel):
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

  est = mle.MLEstimator(lls)
  (x,n,c) = est.maximize(x0,n,c)
  
  assert x<0.01
