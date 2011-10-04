from hyperopt.bandit_algos import GM_BanditAlgo
from hyperopt.bandits import Quadratic1
from idxs_vals_rnd import IndependentAdaptiveParzenEstimator

def test_gm_quadratic1():

    bandit = Quadratic1()
    algo = GM_BanditAlgo(
            good_estimator=IndependentAdaptiveParzenEstimator(),
            bad_estimator=IndependentAdaptiveParzenEstimator(),
            )
    algo.set_bandit(bandit)

