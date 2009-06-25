from brian import *
from brian.library.IF import *

class NeuroToolsSpikeMonitor(SpikeMonitor):
  ''' Spike monitor that outputs a NeuroTools-compatible format '''
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
    header =  "# dt = %s\n" % str(float(self.source.clock.dt*1000))
    header += "# first_id = 0\n"
    header += "# last_id = %i\n" % (len(self.source)-1)
    header += "# dimensions = [1]\n"
    self.f.write(header)

def single_izhikevich_trial(a=0.2/ms,b=0.2/ms,rate=40.0*Hz,deviation=20.0*Hz,time=10000*ms,prefix='results/single_izhikevich_trial'):
  ''' Runs a 1000ms trial of an izhikevich neuron with a single
      randomly rate-varying Poisson input, centered at 40Hz.
      Outputs results of stimulus and neuron firing to:
        prefix/N_in.dat
        prefix/N_out.dat '''
  
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

if(__name__=="__main__"):
  single_izhikevich_trial()
