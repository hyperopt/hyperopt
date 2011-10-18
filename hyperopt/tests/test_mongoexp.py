from hyperopt.base import Bandit, BanditAlgo, Experiment
from hyperopt.mongoexp import MongoExperiment
from hyperopt.bandits import TwoArms
from hyperopt import bandit_algos

def test_MongoExperiment_calls_suggest():
    # I just changed the suggest api in base.Experiment.
    # This test verifies that MongoExperiment.run
    # calls it right.

    # This test does not interact with a database in any way

    class Dummy(MongoExperiment):
        min_queue_len = 10
        poll_interval_secs = 1
        def __init__(self):
            self.queue = []
            self.results = []
            self.trials = []

        def refresh_trials_results(self):
            for config in self.queue:
                self.trials.append(config)
                self.results.append(self.bandit.evaluate(config, None))
            self.queue[:] = []

        def queue_extend(self, configs, skip_dups=True):
            self.queue.extend(configs)
            return configs

        def queue_len(self):
            return len(self.queue)

    d = Dummy()
    d.bandit = TwoArms()
    d.bandit_algo = bandit_algos.Random()
    d.bandit_algo.set_bandit(d.bandit)

    d.run(3)

    print d.queue
