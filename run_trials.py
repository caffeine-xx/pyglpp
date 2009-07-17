from utils import *
from simulations import *
import trials

@print_timing
def run_trial(trial, prefix="experiment", *args):
  (parameters, signal) = getattr(trials, trial)(prefix,*args)
  simu = s.Simulator(**parameters)
  resu = simu.run(signal)
  resu.write_to_file("results/%s_R.pickle" % prefix)
  return resu

if(__name__=="__main__"):
    trial = sys.argv[1]
    prefix = sys.argv[2]
    args = ()
    if len(sys.argv) > 3:
      args = tuple(sys.argv[3:])
    print "=== Running %s -> %s" % (trial,prefix)
    print "Args: ", args
    run_trial(trial,prefix,*args)


