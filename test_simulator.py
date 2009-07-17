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

def two_neuron_trial(prefix="results/two_neuron", perm_input=0, perm_weight=0)
  neur = {'N': 2, 'Ni': 0}
  reco = {'v': True, 'I': True}

  (weight, input) = weight_permutations()

  inputs = input[perm_input]
  weight = weight[perm_weight]
  
  conn = {'weight'=weight}
  inpu = {'weight'=inputs}
  simu = s.Simulator(neurons=neur, record=reco,
                     connect=conn, input=inpu)
  
  time = Trial(t_stop=1.0)
  sign = GaussianNoiseGenerator(30.0, 4.0, 1).generate(time)
   
  resu = simu.run(sign)
  resu.write_to_file("%s_R.pickle" % prefix)
  return resu


if(__name__=="__main__"):
  result = single_izhikevich_trial("results/single_trial_test")

