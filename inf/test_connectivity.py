import numpy as np
import math as m

import numpy.random as rd
import connectivity as c

# Note: this doesn't actually test anything at all, really.
reload(c)


def test_regular_neuron_map():
    nm = c.RegularMap(20,2)
    for i,(x,y) in enumerate(nm.map):
      assert (x%2==0) and (y%2 == 0)

def test_uniform_neuron_map():
    um = c.UniformMap(20,15)
    assert len(um.map)==15

def test_fixed_connectivity():
    nm = c.UniformMap(1000,100)
    fc = c.FixedConnectivity(nm,0.5)
    # assert abs(fc.conn.sum()  - (0.5 * 100**2)) < (0.1 * 100**2)

def test_distance_connectivity():
    nm = c.UniformMap(1000,100)
    dc = c.DistanceConnectivity(nm)
    assert 1 # just hope it don't crash, Watson.

def run_connectivity_tests():
  test_regular_neuron_map()
  test_uniform_neuron_map()
  test_fixed_connectivity()
  test_distance_connectivity()

if __name__ == "__main__":
  run_connectivity_tests()
