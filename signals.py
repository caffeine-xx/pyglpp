from numpy import *
from scipy.ndimage import convolve1d
from NeuroTools import signals

class Trial:
  ''' Defines the timing of a particular trial of an experiment.
      Time is in seconds, by convention.  dt is usually milliseconds.'''
  def __init__(self, t_start=0.0, t_stop=10.0, dt=0.001):
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
    return vectorize(int)((t-self.t_start)/self.dt)

  def duration(self):
    return self.t_stop - self.t_start

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

class SparseBinarySignal(Signal):
  ''' A sparse binary signal is stored as a list of 2ples,
      (time, value) - the value is zero outside of these
      times. This is lazily filled-in when it is first needed. '''
  def __init__(self, trial, signal):
    self.trial  = trial
    self.sparse = signal
    self.signal = None

  def dims(self): return len(self.sparse)
  
  def __call__(self): 
    self.fill() 
    return self.signal

  def fill(self):
    if not getattr(self, 'signal'):
      self.__fill__()
    return self

  def __fill__(self):
    ''' Creates a dense vector of the same signal '''
    
    dims = self.dims()
    result = zeros((dims, self.trial.length()))
    for i in xrange(dims):
      bins = self.trial.time_to_bin(self.sparse[i])
      result[i,bins] = 1
    self.signal = result

  def filter_by(self, filter):
    return self.fill().filter_by(filter)

  def filter_basis(self, filter):
    return self.fill().filter_basis(filter)

class SignalGenerator:
  ''' Signal generators generate a signal for the 
      duration of a trial '''
  def generate(self, trial):
    raise Exception("Not implemented.") 

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

class SineBasisGenerator(SignalGenerator):
  ''' Creates sinusoidal bases as described in:
      Pillow et al., Nature 2009, Supplemental Methods '''
  def __init__(self, a=7, c=1.0):
    ''' a - Parameter determining "width" of curves
        c - Parameter determining translation of curves '''
    self.a = a
    self.c = c

  def generate(self, trial, dim=1):
    phi = lambda j: (j+0.001) * pi / (2)
    dis = lambda t: self.a * log(t + self.c)
    domain = lambda j,t: dis(t) > phi(j) - pi and dis(t) < phi(j) + pi
    basis  = vectorize(lambda j,t: domain(j,t) * 0.5 * (1 + cos(dis(t) - phi(j))))
    result = basis(range(dim), trial.range())
    return Signal(trial, result)

