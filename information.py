import numpy as np
import scipy.stats as st

# mutual information and transfer entropy

def mutual_information(data1, data2):
  calc = lambda e1,e2,ec: e1+e2-ec
  mutinf = np.zeros((len(data1),len(data2)))
  for ii,i in enumerate(data1):
    for jj,j in enumerate(data2):
      pdi = pdf_1d(i)[0]
      pdj = pdf_1d(j)[0]
      pdc = pdf_nd([i,j])[0]
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
  histnd = histnd / histnd.sum()
  values = np.array(range(histnd.size))
  values.reshape(histnd.shape)
  pdf = st.rv_discrete(name=name,values=[values,histnd])
  return (pdf,edges)

def transfer_entropy(ts1, ts2, l=1, bins=5):
  ''' Calculate transfer entropy between two timeseries, with lag l '''
  lts1  = ts1[:-l]
  lts2  = ts2[:-l]
  ts1   = ts1[l:]

  lag1,  bins1   = pdf_1d(lts1,bins=bins)
  lag2,  bins2   = pdf_1d(lts2,bins=bins)

  joint, jedges  = pdf_nd([ts1,  lts1, lts2], bins=[bins1,bins1,bins2])
  lagged,ledges  = pdf_nd([lts1, lts2], bins=[bins1,bins2])
  auto,  aedges  = pdf_nd([ts1,  lts1], bins=[bins1,bins1])

  idx2  = lambda i,j:   i*bins + j
  idx3  = lambda i,j,k: i*(bins**2) + j*bins + k

  jdist = lambda i,j,k: joint.F[idx3(i,j,k)]                     # P(i(t), i(t-1), j(t-1))
  numer = lambda i,j,k: zdiv(jdist(i,j,k) , lagged.F[idx2(j,k)]) # P(i(t) | i(t-1), j(t-1))
  denom = lambda i,j:   zdiv(auto.F[idx2(i,j)] , lag1.F[j])      # P(i(t) | i(t-1))
  trans = np.vectorize(lambda i,j,k: jdist(i,j,k) * zlog(zdiv(numer(i,j,k) , denom(i,j))))

  args  = np.array([(i,j,k) for i in xrange(bins) for j in xrange(bins) for k in xrange(bins)]).T
  sum   = trans(args[0],args[1],args[2])

  return sum.sum()

@np.vectorize
def zlog(h):
  if h==0: return 0
  else: return np.log(h)

@np.vectorize
def zdiv(h1,h2):
  if h2==0: return 0
  else: return np.divide(h1,h2)
