import config
import inference as inf
import result as re
reload(inf)
reload(re)

def main(prefix):

  result = re.load_result(prefix)
  result.import_dict(ml_glpp_parameters(result))
  re.save_result(prefix,result)

  return result

def ml_glpp_parameters(result, model=False):
  ''' Performs ML inference on the given simulation result,
  and returns the resulting parameters '''

  if not model: model = inf.MultiNeuron()

  model.set_data(result.trial.dt, result.trial.length(),
                 result.stimulus.signal, result.spike_trains.sparse_bins())

  K,H,Mu    = model.max_likelihood()
  intensity = model.logI(K,H,Mu)

  parameters = {'K':K, 'H':H, 'Mu':Mu, 'intensity':intensity,
    'stim_basis':model.stim_basis,'spike_basis':model.spike_basis}
  
  return parameters

if(__name__=="__main__"):
  import sys
  prefix = config.results_dir+sys.argv[1]
  print "=== Running analysis: %s" % prefix
  main(prefix)

