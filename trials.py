from utils import *
from numpy import *
from brian import *
from brian.library.IF import *
import cPickle

@print_timing
def run_random_network_trial(prefix,id):
  ''' Runs a trial (a simulation) from trials.py
      Generates parameters from prefix_params, runs prefix_trial,
      and saves the relevant info in results/prefix_id_P.dat '''
  prefix = "results/%s_%i" % (prefix,id)
  params = random_network_params(prefix,id)
  random_network_trial(**params)
  cPickle.dump(params, file(prefix+"_P.pickle", 'w'), 2)

class NeuroToolsSpikeMonitor(SpikeMonitor):
  ''' Spike monitor for Brian that outputs a NeuroTools-compatible format '''
  def __init__(self,source,filename,record=False,delay=0):
      super(NeuroToolsSpikeMonitor,self).__init__(source,record,delay)
      self.filename = filename
      self.f = open(filename,'w')
      self.write_header()
  
  def reinit(self):
      self.close_file()
      self.f = open(self.filename,'w')
      self.write_header()
  
  def propagate(self,spikes):
    super(NeuroToolsSpikeMonitor,self).propagate(spikes)
    for i in spikes:
      self.f.write(str(float(self.source.clock.t*1000))+"\t"+str(i)+"\n")
  
  def write_header(self):
    ''' The whole point of this class.  Stores dt, first_id, last_id,
        and number of dimensions into the header of a raster. '''
    header =  "# dt = %s\n" % str(float(self.source.clock.dt*1000))
    header += "# first_id = 0\n"
    header += "# last_id = %i\n" % (len(self.source)-1)
    header += "# dimensions = [1]\n"
    self.f.write(header)

def single_izhikevich_trial(a=0.2/ms,b=0.2/ms,rate=40.0*Hz,deviation=20.0*Hz,time=10000*ms,prefix='results/single_izhikevich_trial'):
  ''' Runs a trial of an izhikevich neuron with a single
      randomly rate-varying Poisson input, centered at 40Hz.'''
  # Izhikevich model with excitatory synapse
  model  = Izhikevich(a,b)
  reset  = AdaptiveReset(Vr=-75*mV, b=0.2/ms)
  neuron = NeuronGroup(1,model=model, threshold=-30.0*mV, reset=reset)
    
  # Poisson stimulus
  rates           = lambda t: max(0,normal(rate, deviation))
  stimulus        = PoissonGroup(1,rates=rates)
  connection      = Connection(stimulus, neuron)
  connection[0,0] = 40*mvolt
  
  # Spike recording
  in_file  = "%s_S.ras"  % (prefix)
  out_file = "%s_N.ras" % (prefix)
  
  in_monitor  = NeuroToolsSpikeMonitor(stimulus,in_file,record=True)
  out_monitor = NeuroToolsSpikeMonitor(neuron,out_file,record=True)
  
  neuron.clock.reinit()
  run(time)


def random_network_params(prefix="results/random_network_0",id=0):
  ''' For a given id, generates a unique set of parameters for
  a random network trial.  This allows multiple parallel runs of
  trials without fear of overlap - unique IDs mean unique trials. 
    The second argument is the total number of trials.  This allows
  us to scale parameters correctly over the trial space. '''
  S = 1
  N = 2+id
  I = 2
  stimW = rand(S,N) * 10.0*mV
  neurW = rand(N,N) * 10.0*mV * d2_connector(N,range(I),1.2)
  params = {
    'a':0.2 / ms,
    'b':0.2 / ms,
    'rates': "lambda t: normal_rate_generator(t, 60.0*Hz, 20.0*Hz)",
    'S': S,
    'N': N,
    'I': I,
    'stimW': stimW,
    'neurW': neurW,
    'time': 500*ms,
    'prefix': prefix
  }
  return params

def random_network_trial(a=0.2/ms,b=0.2/ms,rates=None,S=20,
                         N=50, I=20, stimW=None, neurW=None,
                         time=1000*ms, prefix='results/random_network_trial'):
  ''' Izhikevich neurons randomly inter-connected, and randomly connected
      a Poisson spiker whose rate has a Normal distribution 
      N - number of neurons, I - number of inhibitory
      S - number of stimuli
      stimW - weight matrix for stimuli
      neurW - weight matrix for neurons '''
  if stimW==None:
    stimW = rand(S,N)*10.0*mvolt * (rand(S,N) < 0.3)
  if neurW==None:
    neurW = rand(N,N)*10.0*mvolt * d2_connector(N,range(I))
  if rates==None:
    rates = normal_rate_generator
  else:
    rates = eval(rates)
  
  # Models 
  model = Izhikevich(a,b)
  reset = AdaptiveReset(Vr=-75*mV, b=0.2/ms)
  neurons  = NeuronGroup(N, threshold=-30.0*mV, model=model, reset=reset)
  stimulus = PoissonGroup(S, rates)
  neurons.vm = rand(N)*20.0*mvolt
  # Connectivity
  stimC = Connection(stimulus, neurons, max_delay=5*ms)
  neurC = Connection(neurons, neurons, max_delay=5*ms)
  stimC = set_connection_voltage(stimC, stimW)
  neurC = set_connection_voltage(neurC, neurW)
  # Recording
  stim_file = "%s_S.ras" % (prefix)
  neur_file = "%s_N.ras" % (prefix)
  stim_out  = NeuroToolsSpikeMonitor(stimulus, stim_file, record=True)
  neur_out  = NeuroToolsSpikeMonitor(neurons, neur_file, record=True)
  # Execute 
  neurons.clock.reinit()
  run(time)

def normal_rate_generator(t, rate=40.0*Hz, deviation=20.0*Hz):
  ''' Generates 'rates' for Poisson spikers according to a 
  Normal distribution '''
  return max(0,normal(rate,deviation))

def d2_connector(N, inhib=[], weight=1.0):
  ''' Connector with probability related to inverted square distance of neurons
      (where 'distance' is a made-up concept based on ID numbers).
      The inhibitory argument lists neuron IDs that are inhibitory, so that
      their weights to other neurons should be negative. 
      Units: [neurons x neurons] * nS '''
  W = zeros((N,N))
  for i in xrange(N):
    for j in xrange(N):
      W[i,j] = rand() < weight*(1.0 - (float((i-j)%N)/float(N))**2)
  W[inhib,:] = -1 * W[inhib,:]
  return W

def set_connection_voltage(conn, weights):
  for i in xrange(len(weights[:,0])):
    for j in xrange(len(weights[0,:])):
      conn[i,j] = weights[i,j]
  return conn

if(__name__=="__main__"):
    prefix = sys.argv[1]
    id = int(sys.argv[2])
    end_id = id+1
    if len(sys.argv) > 3:
      end_id = int(sys.argv[3])+1
    for i in xrange(id, end_id):
      id = i
      print "=== Running trial [%s, %i]" % (prefix,id)
      run_random_network_trial(prefix,id)
