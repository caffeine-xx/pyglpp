import numpy as np
import scipy.stats as st

def information_analysis(dat):
  ''' Calculates mutual info and transfer entropy between:
      - Each pair of neurons (in each direction)
      - The input signal and each neuron
      The neuron signal analyzed is the log-Poisson intensity'''
  bins = 5
  lag = 2
  MI = mutual_information(dat['logI'],dat['logI'],bins=bins)
  TE_X = np.array([[transfer_entropy(x,y,lag=lag,bins=bins) 
                    for y in dat['X']] for x in dat['logI']])
  TE_I = np.array([[transfer_entropy(x,y,lag=lag,bins=bins)
                    for y in dat['logI']] for x in dat['logI']])
  return { 'TE_X':TE_X, 'TE_I':TE_I}

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
      mutinf[ii,jj] = pdi.entropy()+pdj.entropy()-pdc.entropy()
  return mutinf

def pdf_1d(i,bins=4):
  ihist, ibins = np.histogram(i, bins=bins)
  ipdf         = st.rv_discrete(name="i",values=[range(bins),normalize(ihist)])
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
  result = hist.astype(float) / hist.sum(axis=axis)
  return clean(result)

def clean(hist):
  hist[np.isnan(hist)]=0.
  hist[np.isinf(hist)]=0.
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

def transfer_entropy(ts1,ts2,lag=1,bins=4):
  ''' D_1<-2 '''

  ts1,lts1  = multi_lag(ts1,lag)
  ts2,lts2  = multi_lag(ts2,lag)

  # P(i_n+1, i_(n), j_(n))
  joint = np.histogramdd([ts1]+lts1+lts2, bins=bins)[0]
  joint = normalize(joint)
  # P(i_n+1, i_(n))
  auto = np.histogramdd([ts1]+lts1, bins=bins)[0]
  auto = normalize(auto)
  # P(i_(n))
  lag1 = np.histogramdd(lts1,bins=bins)[0]
  lag1 = normalize(lag1)
  # P(i_(n), j_(n))
  lag12 = np.histogramdd(lts1+lts2, bins=bins)[0]
  lag12 = normalize(lag12)
  # P(i_n+1 | i_(n), j_(n))
  jcond = np.divide(joint.T , lag12.T).T
  jcond = clean(jcond) 
  jcond = do_cpdf(jcond.T, avg_zeros).T
  # P(i_n+1 | i_(n))
  acond = np.divide(auto.T, lag1.T).T
  acond = clean(acond)
  acond = do_cpdf(acond.T, avg_zeros).T
  do_cpdf(acond, is_pdf)
  # E log P(i_n+1 | i_(n), j_(n)) / P(i_n+1 | i_(n))
  transfer = joint * clean(np.log(np.divide(jcond , acond)))

  return transfer.sum()

def do_cpdf(pdf, f):
  newpdf = pdf
  if len(pdf.shape)==1:
    newpdf[:] = f(pdf)
  else: 
    for i,j in enumerate(pdf):
      newpdf[i] = do_cpdf(j,f)
  return newpdf

def avg_zeros(pdf):
  ''' Takes a conditional PDF in which the last index is the random var,
      and subsequent indices are conditional vars.  '''
  if pdf.sum() == 0:
    pdf[:] = 1./len(pdf)
  return pdf

def is_pdf(p):
  assert abs(1.0 - p.sum())<0.001
  assert (p>=0).all()
  assert (p<=1).all()
  return True

def poiss_transfer_entropy(params1, params2, stim_filter, spike_filter):

  (K1,H1,Mu1) = params1
  (K2,H2,Mu2) = params2
  

