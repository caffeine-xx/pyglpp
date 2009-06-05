from neuro import *

time = 1000
size = 40
stim = cos_stimulus(time,200)
spikes = spike_train(stim)
times = filter(lambda t: t > size, spike_times(size,spikes))
sf = rand_filter(size)
hf = rand_filter(size)
delta = 1
theta = filt_slice(1, sf, hf)
data = data_slicer(size, stim, spikes)
args = (data, delta, time, size, times)
