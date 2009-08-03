import sys
import config
import result
import trials
from utils import print_timing

@print_timing
def run_trial(trial, prefix="experiment", *args):
  ''' Runs a particular trial.
  Parameters:
    - trial: name of a function in trials.py
    - prefix: file in which the results should be stored.
    - *args: arguments to the trial function '''
  reload(trials)
  resu = getattr(trials, trial)(*args)
  result.save_result(config.results_dir + prefix, resu)
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
  try:
    result.ResultPlotter(resu).plot()
  except:
    print "Couldn't plot: "
