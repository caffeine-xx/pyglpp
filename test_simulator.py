from signals import *
from brian import *
import simulator as s
reload(s)

def single_izhikevich_trial(prefix="results/single_trial_test"):
  neur = {'N': 1, 'Ni': 0}
  reco = {'v': True, 'I': True}
  simu = s.Simulator(neurons=neur, record=reco)
  
  time = Trial(t_stop=2.0)
  sign = FlatlineGenerator(30.0, 1).generate(time)
  resu = simu.run(sign)
   
  resu.write_to_file("%s_R.pickle" % prefix)
  return resu 

def weight_permutations():
  weight = [array([[1.0, 0.0],[0.0, 0.0]])*nS,
            array([[0.0, 1.0],[0.0, 0.0]])*nS,
            array([[0.0, 0.0],[1.0, 0.0]])*nS,
            array([[0.0, 0.0],[0.0, 1.0]])*nS]

  inputs = [array([0.0, 0.0])*nS,
            array([1.0, 0.0])*nS,
            array([1.0, 1.0])*nS]

  return (weight, input)


if(__name__=="__main__"):
  result = single_izhikevich_trial("results/single_trial_test")

