import sys

import simulator as s
import trials
from utils import *

@print_timing
def run_trial(trial, prefix="experiment", *args):
  ''' Runs a particular trial.
  Parameters:
    - trial: name of a function in trials.py
    - prefix: file in which the results should be stored.
    - *args: arguments to the trial function '''
  reload(trials)
  (parameters, signal) = getattr(trials, trial)(prefix,*args)
  simu = s.Simulator(**parameters)
  resu = simu.run(signal)
  resu.write_to_file("results/%s.pickle" % prefix)
  return resu

if(__name__=="__main__"):
  trial = sys.argv[1]
  prefix = sys.argv[2]
  args = ()
  if len(sys.argv) > 3:
    args = tuple(sys.argv[3:])
  print "=== Running %s -> %s" % (trial,prefix)
  print "Args: ", args
  resu = run_trial(trial,prefix,*args)
  resu.plot()
