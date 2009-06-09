import numpy as np
import scipy.optimize as opt
import likelihood_model as lm

class MLEstimator(lm.LikelihoodModel):

  def __init__(self, model):
    self.model = model

  def logL(self,theta, *args):
    a = self.model.unpack(theta, args)
    return -1*self.model.logL(*a)

  def logL_grad(self,theta, *args):
    a = self.model.unpack(theta, args)
    theta, shape = self.model.pack(*tuple(self.model.logL_grad(*a)))
    return -1*theta

  def maximize(self,*a):
    theta, args = self.model.pack(*a)
    theta = opt.fmin_cg(self.logL, theta, self.logL_grad,  args=args)
    return self.model.unpack(theta, args)

"""
  def logL_hess_p(theta, args):
    a = self.model.unpack(theta, args)
    return self.model.logL_hess(*a)
"""
