import hyperopt

class DummyBandit(hyperopt.Bandit):
    param_gen = {"a":10}
    
    def __init__(self):
        super(DummyBandit, self).__init__(self.param_gen)

    def evaluate(config, ctrl):
        raise Exception("Hi!")


def test_failure():
    trials = hyperopt.Trials()
    bandit_algo = hyperopt.Random(DummyBandit())
    exp = hyperopt.Experiment(trials, bandit_algo, async=False)
    try:
        exp.run(0)
    except Exception:
        pass
    else:
        raise Exception("This test should have failed.")
    