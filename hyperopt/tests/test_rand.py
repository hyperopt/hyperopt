
from hyperopt import rand
from hyperopt.tests.test_base import Suggest_API
from hyperopt.bandits import gauss_wave2

TestRand = Suggest_API.make_tst_class(rand.suggest, gauss_wave2(), 'TestRand')

