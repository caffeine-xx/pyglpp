from inference import run_bases
from signals import *
from brian import *
from numpy import *
import simulator as s

reload(s)

def test_lnp_simulator():
  trial = Trial(t_start=0,t_stop=4,dt=1)
  stim  = array([[1,3,5,7]])
  sign  = Signal(trial,stim)

  spike_basis = array([[2,1]])
  stim_basis  = spike_basis
  bst   = run_bases(stim_basis, stim)
  
  N = 2
  K = array([[[1]],[[1]]]) # both neurons driven by input
  H = array([[[0],[0]],[[1],[0]]]) # 0->1, no auto
  M = array([1,1]) # unit mean firing rates
  
  ps = s.LNPSimulator(spike_basis,stim_basis,N,K,H,M)

  bst2  = ps.rebase(stim_basis, stim)
  assert bst.ravel()[-1] == bst2

  resu = ps.run(sign)
  resu.write_to_file("results/lnp_sim_test.pickle")

def test_lnp_big():
  dX = 5
  dY = 5
  dS = 5
  N = 10

  trial = Trial(t_start=0.0, t_stop=1.0, dt=0.001)
  signal = GaussianNoiseGenerator(15.0, 4.0, dS).generate(trial)
  Xb = SineBasisGenerator(7,1,dX).generate(Trial(0.0,1.0,0.1)).signal
  Yb = SineBasisGenerator(7,1,dY).generate(Trial(0.0,1.0,0.1)).signal

  K = random.rand(N,dS,dX)
  H = random.rand(N,N,dY)
  M = random.rand(N)
  
  ps = s.LNPSimulator(Yb,Xb,N,K,H,M)
  re = ps.run(signal)
  re.write_to_file("results/lnp_big_test.pickle")
  print re.monitors

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
#   result = single_izhikevich_trial("results/single_trial_test")
   test_lnp_simulator()
#   test_lnp_big()

