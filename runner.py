import parameters
import trials
import cPickle

def run_trial(prefix, id):
  param_generator = getattr(trials,prefix+"_params")
  params = param_generator(id)
  trial_runner = getattr(trials, prefix+"_trial")
  trial_runner(**params)
  cPickle.Pickler(file("%s_P.dat" % params['prefix'],'w'),2).dump(params)

