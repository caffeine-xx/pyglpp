class Memoize(object):
  """
  Use as a decorator to cache the results of a function.  The function must
  have the following properties:
  * No side effects.
  * Referentially transparent: (see: http://tinyurl.com/reftrans ).
  * No keyword arguments.
  While the following are not required for proper behavior, memoization is
  unlikely to produce a significant performance gain and may indeed produce
  a performance loss unless the function has the following properties:
  * Deep and/or wide recursion.
    -AND/OR-
    Performance slower than a hash lookup on the arguments.
  * The function is called many times with the same arguments.
  """
  def __init__(self,f):
    self.f = f
    self.memo = {}

  def __call__(self,*args):
    if args in self.memo:
        return self.memo[args]
    result = self.f(*args)
    self.memo[args] = result
    return result

class KWMemoize(object):
  """
  This is equivalent to Memoize except that it allows keyword arguments.
  However, there is a small performance cost for this so one should use
  Memoize for functions that do not take keyword arguments.
  """
  def __init__(self,f):
    self.f = f
    self.memo = {}
  def __call__(self,*args,**kwargs):
    kws = tuple(kwargs.items())
    if (args,kws) in self.memo:
        return self.memo[(args,kws)]
    result = self.f(*args,**kwargs)
    self.memo[(args,kws)] = result
    return result

