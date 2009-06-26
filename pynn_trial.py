from numpy import *
from pyNN.brian import *
from pyNN.random import *

def poisson_stim_rand(timestep=1.0, duration=1000.0, stimuli=1, neurons=8, stim_rate=10.0, prefix="results/poisson_stim_rand"):
  """ Input: set of constant-rate Poisson spikers
      Cells: default IF_cond_alpha neurons
      Conns: Input->Cells P=0.25, Cells->Cells P=0.5 """
  
  setup(timestep=timestep, min_delay=1.0, max_delay=1.0)
  
  cells = Population(neurons, IF_cond_alpha, label="cells")
  stims = Population(stimuli, SpikeSourcePoisson, cellparams={'rate':stim_rate}, label="stims")
  cells.record()
  stims.record()
  
  conn_stims = FixedProbabilityConnector(0.25, weights=1.0, allow_self_connections=False)
  proj_stims = Projection(stims, cells, conn_stims, target="excitatory")
  proj_stims.saveConnections("%s_stims.conn" % prefix)
  
  conn_cells = FixedProbabilityConnector(0.5, weights=1.0, allow_self_connections=False)
  proj_cells = Projection(cells, cells, conn_cells, target="excitatory")
  proj_cells.saveConnections("%s_cells.conn" % prefix)
  
  run(duration)
  
  cells.printSpikes("%s_cells.ras" % prefix)
  stims.printSpikes("%s_stims.ras" % prefix)
  
  end()

def poisson_stim_rand_inh(timestep=1.0, duration=1000.0, stimuli=1, neurons=10, inh_neurons=2, stim_rate=10.0, prefix="results/poisson_stim_rand_inh"):
  """ Input: set of constant-rate Poisson spikers
      Cells: default IF_cond_alpha neurons, subset inhibitory
      Conns: Input->Cells P=0.25, Cells->Cells P=0.5 """
  
  rng = NumpyRNG(123)
  uni = RandomDistribution('uniform', [0.0, 20.0], rng)

  setup(timestep=timestep, min_delay=1.0, max_delay=1.0)

  cells_E = Population(neurons, IF_cond_alpha, label="E")
  cells_I = Population(inh_neurons, IF_cond_alpha, label="I")

  cells_E.randomInit(uni)
  cells_I.randomInit(uni)
  
  stims = Population(stimuli, SpikeSourcePoisson, cellparams={'rate':stim_rate}, label="S")

  cells_E.record()
  cells_I.record()
  stims.record()

  conn_stims = FixedProbabilityConnector(0.25, weights=1.0, allow_self_connections=False)
  S_to_E = Projection(stims, cells_E, conn_stims, target="excitatory")
  S_to_I = Projection(stims, cells_I, conn_stims, target="excitatory")
  S_to_E.saveConnections("%s_S_to_E.conn" % prefix)
  S_to_I.saveConnections("%s_S_to_I.conn" % prefix)

  conn_cells   = FixedProbabilityConnector(0.5, weights=0.5, allow_self_connections=False)

  E_to_E = Projection(cells_E, cells_E, conn_cells, target="excitatory")
  E_to_E.saveConnections("%s_E_to_E.conn" % prefix)

  E_to_I = Projection(cells_E, cells_I, conn_cells, target="excitatory")
  E_to_I.saveConnections("%s_E_to_I.conn" % prefix)

  I_to_E = Projection(cells_I, cells_E, conn_cells, target="inhibitory")
  I_to_E.saveConnections("%s_I_to_E.conn" % prefix)

  I_to_I = Projection(cells_I, cells_I, conn_cells, target="inhibitory")
  I_to_I.saveConnections("%s_I_to_I.conn" % prefix)

  run(duration)

  cells_E.printSpikes("%s_E.ras" % prefix)
  cells_I.printSpikes("%s_I.ras" % prefix)
  stims.printSpikes("%s_S.ras" % prefix)

  end()


if (__name__ == "__main__"):
  poisson_stim_rand_inh()
