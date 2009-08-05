import numpy as np
import scipy.stats as st

def lnp_result_info(result):
  ''' For N neurons, results 0-N are lambda monitors, and 
      N+1-2N are inferred intensities '''
  return (information_analysis(result.monitors['lambda']), 
          information_analysis(result.intensity),
          information_analysis(result.intensity,result.monitors['lambda']))

def information_analysis(data1,data2=None):
  ''' Calculates mutual info and transfer entropy between:
      - Each pair of neurons (in each direction)
      - The input signal and each neuron
      The neuron signal analyzed is the log-Poisson intensity'''
  if data2==None: data2 = data1
  MI = do_pairwise(mutual_information, data1, data2)
  TE = do_pairwise(transfer_entropy_pdf, data1, data2)
  return (MI, TE)

def do_pairwise(fun, data1, data2=None):
  if data2==None: data2 = data1
  result = np.zeros((len(data1),len(data2)))
  for i,x in enumerate(data1):
    for j,y in enumerate(data2):
      result[i,j] = fun(x,y)
  return result

def mutual_information(data1, data2, bins=5):
  ''' Calulates mutual information for two
      vectors.  Parameters: vectors data1 & data2
      Returns: mutual information (in nats) '''
  pdi = pdf_1d(data1,bins=bins)[0]
  pdj = pdf_1d(data2,bins=bins)[0]
  pdc = pdf_nd([data1,data2],bins=bins)[0]
  return pdi.entropy()+pdj.entropy()-pdc.entropy()

def pdf_1d(i,bins=4):
  ihist, ibins = np.histogram(i, bins=bins,new=True)
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

def transfer_entropy_pdf(x, y, lag=2, bins=5):
  ''' D_x<-y '''
  x1,xt  = multi_lag(x,lag)
  y1,yt  = multi_lag(y,lag)
  
  # entropies:
  # -X_t + X_t,Y_s + X_t,X_t+1 - X_t,X_t+1,Y_s
  px   = pdf_nd(xt,bins=bins)[0]
  pxy  = pdf_nd(xt+yt,bins=bins)[0]
  pxx  = pdf_nd([x1]+xt,bins=bins)[0]
  pxxy = pdf_nd([x1]+xt+yt,bins=bins)[0]
  
  ex,exy,exx,exxy = map(lambda p: p.entropy(),(px, pxy, pxx, pxxy))
  return exx+exy-ex-exxy

def transfer_entropy(ts1,ts2,lag=2,bins=5):
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
  # E[log P(i_n+1 | i_(n), j_(n)) / P(i_n+1 | i_(n))] 
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
      and subsequent indices are conditional vars.  If all entries
      are zero, turns into a uniform distribution'''
  if pdf.sum() == 0:
    pdf[:] = 1./len(pdf)
  return pdf

def is_pdf(p):
  assert abs(1.0 - p.sum())<0.001
  assert (p>=0).all()
  assert (p<=1).all()
  return True

