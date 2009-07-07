from brian import *
from brian.library.synapses import *
from trials import NeuroToolsSpikeMonitor

C_m = 1.3*pF
Ee  = 0.0*mV
Ei  = -80.0*mV
a   = 0.02
b   = 0.2
c   = -65.0*mV
d   = 8.0*mV
tau_e = 10.0*ms
tau_i = 10.0*ms

eqs = ('dv/dt = ((0.04/mV)*v**2 + 5.0*v + 140.0*mV - w)/ms + I/C_m : volt',
       'dw/dt = a*(b*v - w)/ms : volt',
       'I = ge*(Ee-v) - gi*(Ei-v) : amp',
       'dge/dt = -ge / tau_e : siemens',
       'dgi/dt = -gi / tau_i : siemens'
       )
thr = 'v > 30.0*mV'
res = 'v = c ; w = w + d'
mod = Equations(eqs)
print mod

def big_network():
  neu = NeuronGroup(1000, mod, threshold=thr, reset=res)
  cin = Connection(neu[:200], neu, 'gi', weight=20*nS, sparseness=0.1)
  cex = Connection(neu[200:], neu, 'ge', weight=10*nS, sparseness=0.4)
  inp = PoissonGroup(100, rates=100.0*Hz)
  cnp = Connection(inp, neu, 'ge', weight=10*nS, sparseness=0.3)
  spk = SpikeMonitor(neu, record=True)
  vmm = StateMonitor(neu[290:320], 'v', record=True)
  net = Network(neu, inp, cin, cex, cnp, spk,vmm)
  init(net)
  net.reinit()
  net.run(1000*ms)
  return spk,vmm

def mini_network():
  neu = NeuronGroup(1, mod, threshold=thr,reset=res)
  inp = PoissonGroup(1, rates=100.0*Hz)
  con = Connection(inp, neu, 'ge', weight=10*nS, delay=1*ms)
  spi = SpikeMonitor(neu, record=True)
  vmm = StateMonitor(neu, 'v', record=True)
  imm = StateMonitor(neu, 'I', record=True)
  wmm = StateMonitor(neu, 'w', record=True)
  net = Network(neu, inp, con, vmm, imm, wmm, spi)
  init(net)
  net.reinit()
  net.run(1000*ms)
  return (vmm,imm, wmm,spi)


def init(neu):
  neu.v = c
  neu.w = b * c
  neu.ge = 10*nS
  neu.gi = 20*nS
