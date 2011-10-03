from hyperopt.bandit_algos import GM_BanditAlgo
from hyperopt.bandits import Quadratic1

def test_gm_quadratic1():

    bandit = Quadratic1()
    algo = GM_BanditAlgo()
    algo.set_bandit(bandit)

