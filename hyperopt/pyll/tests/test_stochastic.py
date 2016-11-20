from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import numpy as np
from hyperopt.pyll import scope, as_apply, rec_eval
from hyperopt.pyll.stochastic import recursive_set_rng_kwarg, sample


def test_recursive_set_rng_kwarg():
    uniform = scope.uniform
    a = as_apply([uniform(0, 1), uniform(2, 3)])
    rng = np.random.RandomState(234)
    recursive_set_rng_kwarg(a, rng)
    print(a)
    val_a = rec_eval(a)
    assert 0 < val_a[0] < 1
    assert 2 < val_a[1] < 3


def test_lnorm():
    G = scope
    choice = G.choice
    uniform = G.uniform
    quantized_uniform = G.quniform

    inker_size = quantized_uniform(low=0, high=7.99, q=2) + 3
    # -- test that it runs
    lnorm = as_apply({'kwargs': {'inker_shape': (inker_size, inker_size),
                                 'outker_shape': (inker_size, inker_size),
                                 'remove_mean': choice([0, 1]),
                                 'stretch': uniform(low=0, high=10),
                                 'threshold': uniform(
        low=old_div(.1, np.sqrt(10.)),
        high=10 * np.sqrt(10))
    }})
    print(lnorm)
    print(('len', len(str(lnorm))))
    # not sure what to assert
    # ... this is too fagile
    # assert len(str(lnorm)) == 980


def test_sample_deterministic():
    aa = as_apply([0, 1])
    print(aa)
    dd = sample(aa, np.random.RandomState(3))
    assert dd == (0, 1)


def test_repeatable():
    u = scope.uniform(0, 1)
    aa = as_apply(dict(
        u=u,
        n=scope.normal(5, 0.1),
        l=[0, 1, scope.one_of(2, 3), u]))
    dd1 = sample(aa, np.random.RandomState(3))
    dd2 = sample(aa, np.random.RandomState(3))
    dd3 = sample(aa, np.random.RandomState(4))
    assert dd1 == dd2
    assert dd1 != dd3


def test_sample():
    u = scope.uniform(0, 1)
    aa = as_apply(dict(
        u=u,
        n=scope.normal(5, 0.1),
        l=[0, 1, scope.one_of(2, 3), u]))
    print(aa)
    dd = sample(aa, np.random.RandomState(3))
    assert 0 < dd['u'] < 1
    assert 4 < dd['n'] < 6
    assert dd['u'] == dd['l'][3]
    assert dd['l'][:2] == (0, 1)
    assert dd['l'][2] in (2, 3)
