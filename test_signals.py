from numpy   import *
from signals import *

def test_trial():
  t_start = 0.0
  t_stop  = 10.0
  dt      = 0.1
  length  = (t_stop - t_start) / dt
  t = Trial(t_start, t_stop, dt)
  assert(t.length() == length)
  assert(t.range().size == length)

def test_signal_init():
  x = atleast_2d(ones(10))
  u = Trial(0,11,1)
  try:
    wrong = Signal(u,x)
    assert(False)
  except:
    assert(True)

def test_signal_props():
  x = atleast_2d(ones(10))
  t = Trial(0,10,1)
  s = Signal(t,x)
  assert(1 == s.dims())
  assert(10 == s.length())
  assert(t.length() == s.length())

def test_signal_filtering():
  t = Trial(0,10,1)
  x = Signal(t,atleast_2d(ones(10)))
  u = Trial(0,2,1)
  f = Signal(u,ones(2))
  r = x.filter_by(f)
  assert(r().sum() == 19)
  y = Signal(u,ones((5,2)))
  r = x.filter_basis(y)
  assert(r().shape == (1,10,5))

def test_signal_sparse():
  t = Trial(0,100,0.1)
  d = arange(0, 100, 1.0)
  s = SparseBinarySignal(t,d)
  f = s()
  assert(f.sum() == 100)
  f = Signal(Trial(0,2,1),ones(2))
  x = s.filter_by(f)
  assert(x().sum() == 200)

def test_gaussian_noise():
  t = Trial(0.0,1000,0.1)
  g = GaussianNoiseGenerator(1.0, 3.0)
  s = g.generate(t)
  assert(abs(s().mean()-1.0)<0.5)

def test_sine_bases():
  t = Trial(0.0,10.0,0.1)
  g = SineBasisGenerator()
  s = g.generate(t, 1)
  from matplotlib import pyplot
  pyplot.plot(s())

if (__name__ == "__main__"):
  print test_trial()
  print test_signal_init()
  print test_signal_props()
  print test_signal_filtering()
  print test_signal_sparse()
  print test_gaussian_noise()


