from memoize import *
from numpy import *
from scipy.ndimage import convolve1d
from NeuroTools import signals

class Trial:
  ''' Defines the timing of a particular trial of an experiment.
      Time is in seconds, by convention.'''
  def __init__(self, t_start=0.0, t_stop=5.0, dt=0.001):
    '''  Parameters:
          - t_start = Beginning time.
          - t_stop  = End time.
          - dt      = Timestep '''
    self.t_start = float(t_start)
    self.t_stop  = float(t_stop)
    self.dt      = float(dt)

  def range(self):
    ''' Construct a timeline '''
    return arange(self.t_start, self.t_stop, self.dt)
  
  def length(self):
    ''' Number of bins '''
    return (self.t_stop - self.t_start) / self.dt

  def time_to_bin(self, t):
    ''' Transforms a time interval into the nearest bin '''
    f = vectorize(lambda t: int((float(t)-self.t_start)/self.dt))
    return f(t)

  def duration(self):
    return self.t_stop - self.t_start
  
  def to_hash(self):
    return {
      't_start': self.t_start,
      't_stop':  self.t_stop,
      'dt':      self.dt }

class Signal:
  ''' A signal over a certain period of time, represented as a
      NumPy array. '''
  def __init__(self, trial, signal):
    ''' Parameters:
          - trial  = Specifies the timing of the signal
          - signal = The actual numpy array (or list) specifying the
                     signal itself.  This can be multidimensional. '''
    self.trial  = trial
    self.signal = signal
    if trial.length() != self.length():
      raise Exception("Trial and signal length don't match.")

  def __call__(self):
    return self.signal
  
  def __getitem__(self, t):
    t = self.trial.time_to_bin(float(t))
    if(self.signal.ndim == 1): return signal[t]
    return self.signal[:,t]

  def row(self,key):
    return Signal(self.trial, self.signal[key,:])

  def dims(self):
    return self.signal.ndim==1 and 1 or self.signal.shape[0]

  def length(self):
    return self.signal.ndim==1 and self.signal.size or self.signal.shape[1]

  def fill(self): return self

  def filter_by(self,filter):
    ''' Convolves the rows of this signal with a 1-D signal,
        and returns the resulting signal '''
    if(filter.dims() > 1): raise Exception("Can only filter by a row vector")
    signal = self.signal
    origin = -1 * int(floor(filter.length() / 2))
    result = zeros(signal.shape)
    filter = filter()
    for i in xrange(self.dims()):
        result[i,:] = convolve1d(signal[i,:], filter, mode='constant', cval=0.0, origin = origin)
    return Signal(self.trial, result)

  def filter_basis(self, basis):
    ''' Convolves each row of this signal by each element of
        a basis.  Resulting scheme is [row, time, basis] 
        Parameters
          - basis: Multi-dimensional Signal representing a basis '''
    shape  = tuple(list(self.signal.shape) + [basis.dims()])
    result = zeros(shape)
    for j in xrange(basis.dims()):
      result[:,:,j] = self.filter_by(basis.row(j)).signal
    return Signal(self.trial, result)
  
  def to_analog(self):
    if (self.dims()==1):
      return signals.AnalogSignal(self.signal, **self.trial.to_hash())
    return signals.AnalogSignalList(self.signal, range(self.dims()),  **self.trial.to_hash())

  def plot(self):
    from matplotlib import pylab
    [pylab.plot(x) for x in atleast_2d(self.signal)]

class SparseBinarySignal:

  def __init__(self, trial, signal, multi=False):
    self.trial  = trial
    self.signal = False
    self.bins   = False
    self.sparse = self.break_out(signal)
    self.N      = len(self.sparse)

  def __call__(self):
    return self.fill()()

  def dims(self): return self.N
  
  def break_out(self, signal):
    ''' Breaks a list of (id, time) events into a list of
        [[times], [times]] events '''
    list = [[]]
    for (i,t) in signal:
      while not (i<len(list)): list.append([])
      list[i].append(t)
    return list

  def sparse_bins(self):
    self.bins or self.__bins__()
    return self.bins
  
  def fill(self):
    self.signal or self.__fill__()
    return self.signal
  
  def __bins__(self):
    self.bins = [self.trial.time_to_bin(times) for times in self.sparse]
    
  def __fill__(self):
    ''' Creates a dense vector of the same signal '''
    result = zeros((self.N, self.trial.length()))
    bins = self.sparse_bins()
    for i in xrange(self.N): result[i,bins[i]] = 1
    self.signal = Signal(self.trial,result)
  
  def filter_by(self, filter):
    return self.fill().filter_by(filter)

  def filter_basis(self, filter):
    return self.fill().filter_basis(filter)

  def to_neuro(self):
    return map(lambda s: SpikeTrain(s, t_start = self.trial.t_start, 
                        t_stop = self.trial.t_stop), self.sparse)
   
class SignalGenerator:
  ''' Signal generators generate a signal for the 
      duration of a trial '''
  def generate(self, trial):
    raise Exception("Not implemented.") 

class FlatlineGenerator(SignalGenerator):
  def __init__(self, mean=1.0, dim=1):
    self.mean = mean
    self.dim  = dim

  def generate(self, trial):
    values = self.mean * ones((self.dim, trial.length()))
    return Signal(trial, values)

class GaussianNoiseGenerator(SignalGenerator):
  ''' Generate random Gaussian white noise.'''
  def __init__(self, mean=0.0, std=1.0, dim=1):
    self.mean = mean
    self.std  = std
    self.dim  = dim

  def generate(self,trial):
    noise = random.normal(self.mean, self.std, (self.dim,
      trial.length()))
    return Signal(trial, noise)

class SineWaveGenerator(SignalGenerator):
  def __init__(self, amplitude=1.0, phase=0.0, dim=1):
    self.amplitude = amplitude
    self.phase     = phase
    self.dim       = dim

  def generate(self, trial):
    signal = self.amplitude * sin(trial.range()-self.phase*6.28)
    return Signal(trial, signal)

class SineBasisGenerator(SignalGenerator):
  ''' Creates sinusoidal bases as described in:
      Pillow et al., Nature 2009, Supplemental Methods '''
  def __init__(self, a=7, c=1.0, dim=1):
    ''' a - Parameter determining "width" of curves
        c - Parameter determining translation of curves '''
    self.a = a
    self.c = c
    self.dim = dim

  def generate(self, trial):
    phi = lambda j: (j+0.001) * pi / (2)
    dis = vectorize(lambda t: self.a * log(t + self.c))
    domain = vectorize(lambda j,t: dis(t) > (phi(j) - pi) and dis(t) < (phi(j) + pi))
    basis  = vectorize(lambda j,t: domain(j,t) * 0.5 * (1 + cos(dis(t) - phi(j))))
    result = array([basis(i, trial.range()) for i in range(self.dim)])
    return Signal(trial, result)

