from simulations import *
from signals import *
from brian import *

def single_izhikevich_trial(prefix):
  
  neur = {'N': 1, 'Ni': 0}
  conn = {'weight':0.0*nS}
  inhi = {'weight':0.0*nS}
  simu = Simulator(neurons=neur, connect=conn, inhibit=inhi)
  
  time = Trial(0.0, 1000.0, 1.0)
  sign = GaussianNoiseGenerator(50.0, 30.0, 1).generate(time)
  resu = simu.run(sign)
   
  resu.write_to_file("%s_R.pickle" % prefix)
  return resu

def test_simulator():
  sim = Simulator()
  res = sim.run()


if(__name__=="__main__"):
  single_izhikevich_trial("results/single_trial_test")
