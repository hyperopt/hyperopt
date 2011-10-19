from hyperopt.base import Bandit, BanditAlgo, Experiment
from hyperopt.experiments import SerialExperiment
from hyperopt.bandits import TwoArms
from hyperopt import bandit_algos

def test_SerialExperiment_calls_suggest():
    # I just changed the suggest api in base.Experiment.
    # This test verifies that MongoExperiment.run
    # calls it right.

    d = SerialExperiment(bandit_algos.Random(TwoArms()))

    d.run(3)

    assert len(d.trials) == 3
    assert len(d.results) == 3
