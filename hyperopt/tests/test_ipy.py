import sys
from nose import SkipTest
try:
    from IPython.parallel import Client
except ImportError:
    print >> sys.stderr, "Skipping IPython Tests (IPython not found)"
    raise SkipTest('IPython not present')

from hyperopt.ipy import IPythonTrials
import hyperopt.hp
import hyperopt.tpe
import hyperopt


def simple_objective(args):
    # -- why are these imports here !?
    # -- is it because they need to be imported on the client?
    import time
    import random
    return args ** 2

space = hyperopt.hp.uniform('x', 0, 1)


def test0():
    try:
        client = Client()
    except IOError:
        raise SkipTest()
    trials = IPythonTrials(client)

    minval = trials.fmin(simple_objective, space, hyperopt.tpe.suggest, 25)
    print minval
    assert minval['x'] < .2


def test_fmin_fn():
    try:
        client = Client()
    except IOError:
        raise SkipTest()
    trials = IPythonTrials(client)
    assert not trials._testing_fmin_was_called
    minval = hyperopt.fmin(simple_objective, space,
            algo=hyperopt.tpe.suggest,
            max_evals=25,
            trials=trials)

    assert minval['x'] < .2
    assert trials._testing_fmin_was_called

