require 'mathn'

module Neuro

  def cos_stimulus(time)
    stim = Vector[*(1..time).map{|t| Math.cos( t/24 )}]
  end


  def spike_train(stim)
    spikes = stim.map{|i| (rand < i.abs)?0:1}
  end

  def spike_times(spikes)
    times = []
    spikes.to_a.each_with_index{|s,t| 
      times << t if s == 1
    }
  end

  def filter(t, filt, data)
    data_subset = interval(data, t, t-filt.size())
    filtered = filt.inner_product data_subset
  end

  def interval(vector, t0, t1)
    Vector[*vector.to_a.slice(t0,t1)]
  end

  def log_intensity(t, stim_filter, stim, hist_filter, hist)
    kx = filter(t,stim_filter, stim)
    hy = filter(t,hist_filter, hist)
    intensity = kx + hy
  end

  def log_likelihood(spikes, stim, sf, hf)
    times = spike_times(spikes)
    intensities = (1..times.size()).map{|t| log_intensity(t, sf, stim, hf, spikes)}
    intensities.sum 
  end
  


end

  b = stimulus(1000)
  puts b.inspect

  s = spike_train(b)
  puts s.inspect

  c = train_to_counts(s)
  puts c.inspect
