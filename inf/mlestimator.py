import numpy as np
import scipy.optimize as opt
import likelihood_model as lm

class MLEstimator(lm.LikelihoodModel):

  def __init__(self, model):
    self.model = model

  def logL(theta, args):
    a = self.model.unpack(theta, args)
    return self.model.logL(*a)

  def logL_grad(theta, args):
    a = self.model.unpack(theta, args)
    return self.model.logL_grad(*a)

  def logL_hess_p(theta, args):
    a = self.model.unpack(theta, args)
    return self.model.logL_hess(*a)

  def maximize(*a):
    theta, args = self.model.pack(*a)
    theta = opt.fmin_ncg(-1*logL, theta, -1*logL_grad, -1*logL_hess_p, args=args)
    return self.model.unpack(theta, args)

