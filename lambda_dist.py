from numpy import *
from scipy import weave
from matplotlib import pyplot
import signals

# Toy file to generate a distribution over lambdas 

bas = signals.Trial(0.0,2*pi,dt=pi/4.0)
siz = 2*bas.length()
dim = 4
spike = signals.SineBasisGenerator(2,dim=dim).generate(bas).signal
stim  = signals.SineBasisGenerator(2,dim=dim).generate(bas).signal
parm  = random.randn(dim)*0.3+0.4

full = zeros((dim,siz))
full[:,:(siz/2)] = stim
full[:,(siz/2):] = spike

maxi = 2**(siz)
lamb = zeros(maxi)

for k in xrange(maxi):
  indices = filter(lambda u: k&(2**u),xrange(siz))
  lamb[k] = dot(parm,full[:,indices].sum(1))

pyplot.figure()
pyplot.plot(histogram(lamb, bins=60, normed=True)[0])

pyplot.figure()
pyplot.plot(histogram(exp(lamb), bins=80, normed=True)[0])

pyplot.show()
