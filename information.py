import numpy as np
import scipy.stats as st

# mutual information and transfer entropy

def mutual_information(data1, data2, bins=10):
  ''' Calulates a matrix of pairwise mutual informations for two
      matrices.  Parameters:
        - data1: a PxN1 matrix
        - data2: a QxN2 matrix
      Returns: a PxQ matrix of mutual information (in nats) '''
  calc = lambda e1,e2,ec: e1+e2-ec
  mutinf = np.zeros((len(data1),len(data2)))
  for ii,i in enumerate(data1):
    for jj,j in enumerate(data2):
      pdi = pdf_1d(i,bins=bins)[0]
      pdj = pdf_1d(j,bins=bins)[0]
      pdc = pdf_nd([i,j],bins=bins)[0]
      mutinf[ii,jj] = calc(pdi.entropy(),pdj.entropy(),pdc.entropy())
  return mutinf

def pdf_1d(i,bins=4):
  ihist, ibins = np.histogram(i, normed=True,bins=bins)
  ipdf         = st.rv_discrete(name="i",values=[range(bins),ihist])
  return ipdf,ibins

def pdf_nd(arr, bins=4, name="nd"):
  ''' Produces an N-dimensional joint pdf like P(X1=x1, X2=x2,...) 
      Parameters
        - arr: NxM matrix, where N is dimensions and M is data points
      Returns: (pdf, edges) where:
        - pdf: discrete PDF (indexed by bins)
        - edges: bin edges for discrete PDF values '''
  histnd,edges = np.histogramdd(arr,bins=bins)
  histnd = normalize(histnd)
  values = np.array(range(histnd.size))
  values.reshape(histnd.shape)
  pdf = st.rv_discrete(name=name,values=[values,histnd])
  return (pdf,edges)

def marginalize(hist, axis=0):
  return np.sum(hist, axis=axis)

def normalize(hist,axis=None):
  return hist / hist.sum(axis=axis)

def clean(hist):
  hist[np.isnan(hist)]=0
  hist[np.isinf(hist)]=0
  return hist

def h2pdf(hist,axis=None):
  return clean(normalize(hist,axis=axis))

def multi_lag(x, L=1):
  ''' Generates clipped, lagged timeseries from the original x.
  Returns a tuple containing:
    - newest: unlagged timeseries 
    - lagged: list of L timeseries, where lagged[i] is
              L-i timesteps lag '''
  newest = np.roll(x,-L)[:-L]
  lagged = [np.roll(x,-l)[:-L] for l in range(L)]
  return newest,lagged

def multi_marginalize(x, axes=[0]):
  ''' Marginalizes along multiple axes'''
  result = x
  ax = axes
  ax.reverse()
  for a in ax:
    result = marginalize(result, a)
  return result

def multi_transfer_entropy(ts1,ts2,lag=1,bins=10):
  ts1,lts1  = multi_lag(ts1,lag)
  ts2,lts2  = multi_lag(ts2,lag)
  lts2_ax   = range(lag+1,2*lag+1)

  lag1   = h2pdf(np.histogramdd(lts1,bins=bins)[0])
  joint  = h2pdf(np.histogramdd([ts1]+lts1+lts2, bins=bins)[0])
  lagged = h2pdf(marginalize(joint, 0))
  auto   = h2pdf(multi_marginalize(joint, lts2_ax))
  jcond  = h2pdf(np.true_divide(joint , lagged),axis=0)
  acond  = h2pdf(np.true_divide(auto , lag1),axis=0)

  logratio  = clean(np.log(np.true_divide(jcond , acond)))
  transfer = clean(joint * logratio)
  return transfer.sum()

def transfer_entropy(ts1, ts2, l=1, bins=10):
  ''' Calculate transfer entropy between two timeseries, with lag l '''
  lts1  = ts1[:-l]
  lts2  = ts2[:-l]
  ts1   = ts1[l:]

  lag1   = h2pdf(np.histogramdd([lts1],bins=bins)[0])
  joint  = h2pdf(np.histogramdd([ts1,  lts1, lts2], bins=bins)[0])
  lagged = h2pdf(marginalize(joint, 0))
  auto   = h2pdf(marginalize(joint, 2))
  jcond  = h2pdf(np.true_divide(joint , lagged),axis=0)
  acond  = h2pdf(np.true_divide(auto , lag1),axis=0)

  logratio  = clean(np.log(np.true_divide(jcond , acond)))
  transfer = clean(joint * logratio)
  return transfer.sum()

