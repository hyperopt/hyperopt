import unittest
import numpy as np

from pyll import as_apply, scope, rec_eval, clone, dfs
from pyll.stochastic import replace_implicit_stochastic_nodes
from pyll.stochastic import replace_repeat_stochastic

from hyperopt.vectorize import VectorizeHelper
from hyperopt.vectorize import node_names

def test_replace_implicit_stochastic_nodes():
    a = scope.uniform(-2, -1)
    rng = np.random.RandomState(234)
    new_a, lrng = replace_implicit_stochastic_nodes(a, rng)
    print new_a
    assert new_a.name == 'getitem'
    assert new_a.pos_args[0].name == 'draw_rng'


def test_replace_implicit_stochastic_nodes_multi():
    uniform = scope.uniform
    a = as_apply([uniform(0, 1), uniform(2, 3)])
    rng = np.random.RandomState(234)
    new_a, lrng = replace_implicit_stochastic_nodes(a, rng)
    print new_a
    val_a = rec_eval(new_a)
    assert np.allclose(val_a, (0.03096734347001351, 2.254282073234248))


def config0():
    p0 = scope.uniform(0, 1)
    p1 = scope.uniform(2, 3)
    p2 = scope.one_of(-1, p0)
    p3 = scope.one_of(-2, p1)
    p4 = 1
    p5 = [3, 4, p0]
    d = locals()
    del d['p1'] # -- don't sample p1 all the time
    s = as_apply(d)
    return s


def config1():
    p0 = scope.uniform(0, 1)
    return as_apply(locals())


def test_clone():
    config = config0()
    config2 = clone(config)

    nodeset = set(dfs(config))
    assert not any(n in nodeset for n in dfs(config2))
    
    r = rec_eval(
            replace_implicit_stochastic_nodes(
                config,
                scope.rng_from_seed(5))[0])
    print r
    r2 = rec_eval(
            replace_implicit_stochastic_nodes(
                config2,
                scope.rng_from_seed(5))[0])

    print r2
    assert r == r2


def test_vectorize_config0():
    config = config0()
    assert 'p3' == config.named_args[2][0]
    p1 = config.named_args[2][1].pos_args[1]
    assert p1.name == 'uniform'
    assert p1.pos_args[0]._obj == 2
    assert p1.pos_args[1]._obj == 3

    N = as_apply(5)
    expr = config
    expr_idxs = scope.range(N)
    vh = VectorizeHelper(expr, expr_idxs)
    vh.build_idxs()
    vh.build_vals()
    vconfig = vh.vals_memo[expr]

    full_output = as_apply([vconfig, vh.idxs_by_id(), vh.vals_by_id()])

    print '=' * 80
    print 'VECTORIZED'
    print full_output
    print '\n' * 1

    fo2 = replace_repeat_stochastic(full_output)
    print '=' * 80
    print 'VECTORIZED STOCHASTIC'
    print fo2
    print '\n' * 1

    new_vc, lrng = replace_implicit_stochastic_nodes(
            fo2,
            as_apply(np.random.RandomState(1))
            )

    print '=' * 80
    print 'VECTORIZED STOCHASTIC WITH RNGS'
    print new_vc

    foo, idxs, vals = rec_eval(new_vc)

    print foo
    print idxs
    print vals
    assert len(foo) == 5
    assert foo[0] == {
            'p0': 0.12812444792935673,
            'p2': 0.12812444792935673,
            'p3': -2,
            'p4': 1,
            'p5': (3, 4, 0.12812444792935673)}

    assert foo[1] ==  {
            'p0': 0.99904051532414473,
            'p2': 0.99904051532414473,
            'p3': -2,
            'p4': 1,
            'p5': (3, 4, 0.99904051532414473)}

    assert foo[2]['p3'] != -2
    

    print idxs[vh.node_id[p1]]
    print vals[vh.node_id[p1]]

    assert list(idxs[vh.node_id[p1]]) == [2]
    for ii in range(5):
        if ii in idxs[vh.node_id[p1]]:
            assert foo[ii]['p3'] == vals[vh.node_id[p1]][list(idxs[vh.node_id[p1]]).index(ii)]
        else:
            assert foo[ii]['p3'] == -2, foo[ii]['p3']

