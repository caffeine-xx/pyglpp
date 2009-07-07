from simulations import *
from signals import *
from brian import *

def test_simulator():
  sim = Simulator()
  res = sim.run()
  
eqs = '''dv/dt = ((0.04/mV)*v**2 + 5.0*v + 140.0*mV - w)/ms + I/C_m : volt
        dw/dt = a*(b*v - w)/ms : volt
        I = ge*(Ee-v) - gi*(Ei-v) : amp
        dge/dt = -ge / te : siemens
        dgi/dt = -gi / ti : siemens'''

args = {'a':0.02,     'b':0.2, 
                          'c':-65.0*mV, 'd':8.0*mV,
                          'Ee':0*mV,    'Ei':-80.0*mV, 
                          'te':10*ms,   'ti':10*ms,
                          'C_m':1.3*pF}
noise = GaussianNoiseGenerator(100.0, 10.0, 40)
sig   = noise.generate(Trial(0.1, 100.0, 0.1))
spker = PoissonGroup(40, rates = lambda t: sig[t])
moni  = SpikeMonitor(spker, record=True)
run(sig.trial.duration()*ms)
print moni.spikes

#test_simulator()
