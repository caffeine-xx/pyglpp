from os import path
import result as r
reload(r)

def test_plot():
  p = 'results/single_trial_test_R.pickle'
  assert(path.exists(p))
  res = r.load_result(p)
  res.plot()

if(__name__=="__main__"):
  test_plot()
