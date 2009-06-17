#! /usr/bin/env python

"""
Plot several runs of the iaf_cond_exp_sfa_rr neuron with no input and
various initial values for the membrane potential.
"""

import nest 
import numpy
import pylab

for vinit in numpy.arange(-100, -50, 10, float):

    nest.ResetKernel()

    cbn = nest.Create('iaf_cond_exp_sfa_rr')

    # set the initial membrane potential
    nest.SetStatus(cbn, 'V_m', vinit)

    voltmeter = nest.Create('voltmeter')
    nest.SetStatus(voltmeter, {'withtime': True})
    nest.Connect(voltmeter, cbn)

    nest.Simulate(75.0)

    t = nest.GetStatus(voltmeter,"events")[0]["times"]
    v = nest.GetStatus(voltmeter,"events")[0]["potentials"]

    pylab.plot(t, v, label="initial V_m=%.2f mV" % vinit)

pylab.legend(loc=4)
pylab.xlabel("time (ms)")
pylab.ylabel("V_m (mV)")
pylab.show()
