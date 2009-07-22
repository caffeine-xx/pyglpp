from numpy import *
import information as inf
reload(inf)

def test_mutual_information():
  x1 = random.randn(1,100)
  x2 = random.randn(1,100)
  x3 = x1*2.8+random.randn(1,100)*0.3
  e12 = inf.mutual_information(x1,x2,bins=4)
  e13 = inf.mutual_information(x1,x3,bins=4)
  assert e13 > e12

def test_transfer_entropy():
  x1 = random.randn(1000)
  x2 = roll(x1,2)*2.7+random.randn(1000)*0.2
  x1,x2 = x1[:-1],x2[:-1]
  e21 = inf.transfer_entropy(x2,x1,lag=2,bins=4)
  e12 = inf.transfer_entropy(x1,x2,lag=2,bins=4)
  assert e21 > e12

test_mutual_information()
test_transfer_entropy()

