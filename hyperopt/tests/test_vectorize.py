import unittest
import numpy as np

from pyll import as_apply, scope, rec_eval, clone, dfs, clone_merge
from pyll.stochastic import recursive_set_rng_kwarg

from hyperopt import base
from hyperopt.vectorize import VectorizeHelper
from hyperopt.vectorize import replace_repeat_stochastic
from hyperopt.pyll_utils import PBandit
from hyperopt.pyll_utils import hp_choice
from hyperopt.pyll_utils import hp_uniform
from hyperopt.pyll_utils import hp_quniform
from hyperopt.pyll_utils import hp_loguniform
from hyperopt.pyll_utils import hp_qloguniform


def config0():
    p0 = scope.uniform(0, 1)
    p1 = scope.uniform(2, 3)
    p2 = scope.one_of(-1, p0)
    p3 = scope.one_of(-2, p1)
    p4 = 1
    p5 = [3, 4, p0]
    p6 = scope.one_of(-3, p1)
    d = locals()
    #d['p1'] = None # -- don't sample p1 all the time, only if p3 says so
    s = as_apply(d)
    return s


def test_clone():
    config = config0()
    config2 = clone(config)

    nodeset = set(dfs(config))
    assert not any(n in nodeset for n in dfs(config2))

    foo = recursive_set_rng_kwarg(
                config,
                scope.rng_from_seed(5))
    r = rec_eval(foo)
    print r
    r2 = rec_eval(
            recursive_set_rng_kwarg(
                config2,
                scope.rng_from_seed(5)))

    print r2
    assert r == r2


def test_vectorize_trivial():
    N = as_apply(15)

    p0 = hp_uniform('p0', 0, 1)
    loss = p0
    print loss
    expr_idxs = scope.range(N)
    vh = VectorizeHelper(loss, expr_idxs, build=True)
    vloss = vh.vals_memo[loss]

    full_output = as_apply([vloss,
        vh.idxs_by_id(),
        vh.vals_by_id()])
    fo2 = replace_repeat_stochastic(full_output)

    new_vc = recursive_set_rng_kwarg(
            fo2,
            as_apply(np.random.RandomState(1)),
            )

    #print new_vc
    losses, idxs, vals = rec_eval(new_vc)
    print 'losses', losses
    print 'idxs p0', idxs['p0']
    print 'vals p0', vals['p0']
    p0dct = dict(zip(idxs['p0'], vals['p0']))
    for ii, li in enumerate(losses):
        assert p0dct[ii] == li


def test_vectorize_simple():
    N = as_apply(15)

    p0 = hp_uniform('p0', 0, 1)
    loss = p0 ** 2
    print loss
    expr_idxs = scope.range(N)
    vh = VectorizeHelper(loss, expr_idxs, build=True)
    vloss = vh.vals_memo[loss]

    full_output = as_apply([vloss,
        vh.idxs_by_id(),
        vh.vals_by_id()])
    fo2 = replace_repeat_stochastic(full_output)

    new_vc = recursive_set_rng_kwarg(
            fo2,
            as_apply(np.random.RandomState(1)),
            )

    #print new_vc
    losses, idxs, vals = rec_eval(new_vc)
    print 'losses', losses
    print 'idxs p0', idxs['p0']
    print 'vals p0', vals['p0']
    p0dct = dict(zip(idxs['p0'], vals['p0']))
    for ii, li in enumerate(losses):
        assert p0dct[ii] ** 2 == li


def test_vectorize_multipath():
    N = as_apply(15)

    p0 = hp_uniform('p0', 0, 1)
    loss = hp_choice('p1', [1, p0, -p0]) ** 2
    expr_idxs = scope.range(N)
    vh = VectorizeHelper(loss, expr_idxs, build=True)

    vloss = vh.vals_memo[loss]
    print vloss

    full_output = as_apply([vloss,
        vh.idxs_by_id(),
        vh.vals_by_id()])

    losses, idxs, vals = rec_eval(full_output)
    print 'losses', losses
    print 'idxs p0', idxs['p0']
    print 'vals p0', vals['p0']
    print 'idxs p1', idxs['p1']
    print 'vals p1', vals['p1']
    p0dct = dict(zip(idxs['p0'], vals['p0']))
    p1dct = dict(zip(idxs['p1'], vals['p1']))
    for ii, li in enumerate(losses):
        if p1dct[ii]:
            assert li == p1dct[ii] ** 2
        else:
            assert li == 1

def test_vectorize_config0():
    config = config0()
    assert 'p3' == config.named_args[3][0]
    print config.named_args
    assert config.named_args[3][1].name == 'switch'
    p1 = config.named_args[3][1].pos_args[2]
    assert p1.name == 'uniform'
    assert p1.pos_args[0]._obj == 2
    assert p1.pos_args[1]._obj == 3

    N = as_apply(5)
    expr = config
    expr_idxs = scope.range(N)
    vh = VectorizeHelper(expr, expr_idxs, build=True)
    vconfig = vh.vals_memo[expr]

    full_output = as_apply([vconfig, vh.idxs_by_id(), vh.vals_by_id()])

    if 0:
        print '=' * 80
        print 'VECTORIZED'
        print full_output
        print '\n' * 1

    fo2 = replace_repeat_stochastic(full_output)
    if 0:
        print '=' * 80
        print 'VECTORIZED STOCHASTIC'
        print fo2
        print '\n' * 1

    new_vc = recursive_set_rng_kwarg(
            fo2,
            as_apply(np.random.RandomState(1))
            )
    if 0:
        print '=' * 80
        print 'VECTORIZED STOCHASTIC WITH RNGS'
        print new_vc

    foo, idxs, vals = rec_eval(new_vc)

    print 'foo[0]', foo[0]
    print 'foo[1]', foo[1]
    #print idxs
    #print vals
    assert len(foo) == 5
    assert foo[0] == {
            'p0': 0.39676747423066994,
            'p1': None,
            'p2': 0.39676747423066994,
            'p3': 2.1281244479293568,
            'p4': 1,
            'p5': (3, 4, 0.39676747423066994) }
    assert foo[1] != foo[2]

    if 0:
        print idxs[vh.node_id[p1]]
        print vals[vh.node_id[p1]]

    # - p1 is only used sometimes
    assert len(idxs[vh.node_id[p1]]) < 5
    for ii in range(5):
        if ii in idxs[vh.node_id[p1]]:
            assert foo[ii]['p3'] == vals[vh.node_id[p1]][list(idxs[vh.node_id[p1]]).index(ii)]
        else:
            assert foo[ii]['p3'] == -2, foo[ii]['p3']


def test_distributions():
    # test that the distributions come out right

    # XXX: test more distributions
    bandit = PBandit(as_apply([
                hp_loguniform('lu', -2, 2),
                hp_qloguniform('qlu', np.log(1 + 0.01), np.log(20), 2),
                hp_quniform('qu', -4.999, 5, 1),
                hp_uniform('u', 0, 10)])
            )
    algo = base.Random(bandit)
    trials = base.Trials()
    exp = base.Experiment(trials, algo)
    exp.catch_bandit_exceptions = False
    N = 1000
    exp.run(N)
    assert len(trials) == N
    idxs, vals = base.miscs_to_idxs_vals(trials.miscs)
    print idxs.keys()

    # -- loguniform
    log_lu = np.log(vals[bandit.node_id('lu', algo)])
    assert len(log_lu) == N
    assert -2 < np.min(log_lu)
    assert np.max(log_lu) < 2
    h = np.histogram(log_lu)[0]
    print h
    assert np.all(75 < h)
    assert np.all(h < 125)

    # -- quantized log uniform
    qlu = vals[bandit.node_id('qlu', algo)]
    assert np.all(np.fmod(qlu, 2) == 0)
    assert np.min(qlu) == 2
    assert np.max(qlu) == 20
    bc_qlu = np.bincount(qlu)
    assert bc_qlu[2] > bc_qlu[4] > bc_qlu[6] > bc_qlu[8]

    # -- quantized uniform
    qu = vals[bandit.node_id('qu', algo)]
    assert np.min(qu) == -5
    assert np.max(qu) == 5
    assert np.all(np.fmod(qu, 1) == 0)
    bc_qu = np.bincount(np.asarray(qu).astype('int') + 5)
    assert np.all(40 < bc_qu), bc_qu  # XXX: how to get the distribution flat
    # with new rounding rule?
    assert np.all(bc_qu < 125), bc_qu

    # -- uniform
    u = vals[bandit.node_id('u', algo)]
    assert np.min(u) > 0
    assert np.max(u) < 10
    h = np.histogram(u)[0]
    print h
    assert np.all(75 < h)
    assert np.all(h < 125)


    #import matplotlib.pyplot as plt
    #plt.hist(np.log(vals['node_2']))
    #plt.show()



