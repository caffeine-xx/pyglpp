import time
from brian import SpikeMonitor

def print_timing(func):
  def wrapper(*arg):
    t1 = time.time()
    res = func(*arg)
    t2 = time.time()
    print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
    return res
  return wrapper

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
