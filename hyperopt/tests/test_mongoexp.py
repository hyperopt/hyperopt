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
        def __init__(self, a):
            Experiment.__init__(self, a)
            self.queue = []
            self.results = []
            self.trials = []

        def refresh_trials_results(self):
            for config in self.queue:
                self.trials.append(config)
                self.results.append(
                        self.bandit_algo.bandit.evaluate(config, None))
            self.queue[:] = []

        def queue_extend(self, configs, skip_dups=True):
            self.queue.extend(configs)
            return configs

        def queue_len(self):
            return len(self.queue)

    d = Dummy(bandit_algos.Random(TwoArms()))
    d.bandit = TwoArms()

    d.run(3)
    d.refresh_trials_results()

    assert len(d.queue) == 0
    assert len(d.trials) == 3
    assert len(d.results) == 3
