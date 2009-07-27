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

  if not model:
    model = inf.MultiNeuron()

  model.set_data(result.trial.dt, result.trial.length(), result.input.signal, result.output.signal)
  K,H,Mu    = model.max_likelihood()
  intensity = model.logI(K,H,Mu)

  parameters = {'K':K, 'H':H, 'Mu':Mu, 'intensity':intensity,
    'stim_basis':model.stim_basis,'spike_basis':model.spike_basis}
  
  return parameters

if(__name__=="__main__"):
  import sys

  prefix = sys.argv[1]

  prefs = ["results/"+prefix] 
  id = 0
  end_id = 0

  if(len(sys.argv)>2):
    id = int(sys.argv[2])
    end_id = id+1
    prefs = []
  if len(sys.argv)>3:
    end_id = int(sys.argv[3])+1

  if not id==0:
    prefs = ["results/%s_%i" % (prefix, i) for i in range(id, end_id)]

  for p in prefs:
    print "=== Running analysis: %s" % p
    main(p)
