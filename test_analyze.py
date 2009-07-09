import analyze as a
reload(a)

def test_analyze():
  prefix = "results/single_trial_test"
  res = a.run_analysis(prefix)
  print res

if (__name__ == "__main__"):
  print test_analyze()

