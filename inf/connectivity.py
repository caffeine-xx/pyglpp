import numpy as np
import numpy.random as rd

class NeuronMap:
  """ List of neuron positions """
  def __init__(self, *args):
    self.map           = self.place(*args)
    self.dist          = self.__calc_dist()
    self.N             = len(self.map)

  def __calc_dist(self):
    N = len(self.map)
    dist = np.zeros((N,N))
    for i,(x,y) in enumerate(self.map):
      for j,(x2,y2) in enumerate(self.map):
        dist[i,j] = np.hypot(x-x2,y-y2)
    return dist

  def place(self):
    raise Exception("Not implemented")

class RegularMap(NeuronMap):
  """ A regular map places neurons at completely regular intervals. """

  def place(self,size,d):
    xpos   = range(0,size,d)
    ypos   = range(0,size,d)
    self.N = len(xpos) * len(ypos)
    map    = list()
    for i in xpos:
      for j in ypos:
        map.append((i,j))
    return map

class UniformMap(NeuronMap):
  """ Uses a uniform random distribution of X coords and Y coords """

  def place(self,size,N):
    """ density - coordinates must be a multiple of this to have a neuron """
    xpos = rd.random_integers(1,size,N)
    ypos = rd.random_integers(1,size,N)
    map = zip(xpos,ypos)
    return map

class Connectivity:
  """ Connectivity of a map """
  def __init__(self,map,*args):
    self.map  = map.map
    self.N    = map.N
    self.dist = map.dist
    self.conn = np.zeros((self.N,self.N))
    self.conn = self.connect(*args)

  def connect(self,*args):
    raise Exception("Not implemented")

class FixedConnectivity(Connectivity):
  """ Defines a bernoulli connectivity regime """

  def connect(self,p):
    conn = self.conn
    for i,(ix,iy) in enumerate(self.map):
      for j,(jx,jy) in enumerate(self.map):
        conn[i,j] = (p < rd.rand())
    return conn

class DistanceConnectivity(Connectivity):
  """ Bernoulli connects based on the inverse square of distance """

  def __dist_fn(self,d):
    return 1 - (d / (self.dist.max()))**2

  def connect(self):
    conn = self.conn
    for i,(ix,iy) in enumerate(self.map):
      for j,(jx,jy) in enumerate(self.map):
        d = self.dist[i,j]
        conn[i,j] = (rd.rand() < self.__dist_fn(d))
    return conn
