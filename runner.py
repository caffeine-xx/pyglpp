import sys
from trials import *
from analyze import *
from numpy import *
import cPickle

def calc_filters(params):
  Xfilters = zeros((params['N']+params['I'],params['S'],params['Xb'].shape[1]))
  Yfilters = zeros((params['N']+params['I'],params['N']+params['I'],params['Yb'].shape[1]))
  for i in xrange(params['N']):
    for j in xrange(params['S']):
      Xfilters[i,j,:] = np.average(params['Xb'],axis=0,weights=params['K'][i,j,:])
    for j in xrange(params['N']):
      Yfilters[i,j,:] = average(params['Yb'],axis=0,weights=params['H'][i,j,:])
  return (Xfilters,Yfilters)

def load_results(prefix,id):
  filename = "%s_%i" %(prefix,id)
  params = cPickle.load(file(filename+"_P.pickle"))
  params.update(io.loadmat(filename+"_R.mat"))
  params['K'] = params['K'].reshape((params['N'],params['S'],params['Xb'].shape[0]))
  params['H'] = params['H'].reshape((params['N'],params['N'],params['Yb'].shape[0]))
  params.update(zip(('dt','T','stim','spike'),load_brian_experiment(filename)))
  params['filename'] = filename
  return params
