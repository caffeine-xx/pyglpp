
import numpy as np
import math as m

import numpy.random as rd
import connectivity as c
import nest_simulator as ns

# Note: this doesn't actually test anything at all, really.
reload(c)
reload(ns)


def test_nest_sim():
    nm = c.UniformMap(1000,100)
    fc = c.FixedConnectivity(nm,0.2)
    si = ns.NestSimulator(0.1, "", fc)
    print nm.dist
    return si.run(1000)

