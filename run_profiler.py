import hotshot, hotshot.stats, hotshot.Profile
import test_runner
prof = hotshot.Profile("runner.prof")
time, result = prof.runcall(test_runner.test_analyze)
prof.close()


stats = hotshot.stats.load("runner.prof")
stats.strip_dirs()
stats.sort_stats('time','calls')
stats.print_stats(20)


