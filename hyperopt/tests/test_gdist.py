import hyperopt.gdist as gd
import theano
import numpy as np
from numpy import array, int32


def base_func(s, length, names, random_types, idxs, vals, res):
    template = gd.gDist(s)

    random_nodes = template.random_nodes()

    assert len(random_nodes) == length
    assert template.flatten_names() == names
    assert [node.__class__.__name__ for node in random_nodes] == random_types
    assert res == template.idxs_vals_to_dict_list(idxs, vals)


def test1():
    s = '{"p0" : uniform(0,1)}'
    idxs = (np.array([0]),)
    vals = ([np.array([0.72])],)
    res = [{"p0":vals[0][0][0]}]
    base_func(s, 1, ['conf', 'conf.p0'], ['gUniform'], idxs, vals, res)


def test1a():
    s = '{"p0" : gaussian(0,1)}'
    idxs = (np.array([0]),)
    vals = ([np.array([0.72711124])],)
    res = [{"p0":vals[0][0][0]}]
    base_func(s, 1, ['conf', 'conf.p0'], ['gGauss'], idxs, vals, res)


def test1b():
    s = '{"p0" : choice([0, 1])}'
    idxs = (np.array([0]),)
    vals = ([np.array([1])],)
    res = [{"p0":vals[0][0][0]}]
    base_func(s, 1, ['conf', 'conf.p0'], ['gChoice'], idxs, vals, res)


def test2():
    s = '{"p0" : uniform(0, 1),"p1": gaussian(0, 1)}'
    idxs = (np.array([0]), np.array([0]))
    vals = (np.array([0.92]), np.array([0.73666226]))
    res = [{'p0': vals[0][0], 'p1': vals[1][0]}]
    base_func(s, 2, ['conf', 'conf.p0', 'conf.p1'], ['gUniform', 'gGauss'],
              idxs, vals, res)


def test3():
    s = '{"p0" : uniform(0,1) + gaussian(0,1)}'
    idxs = (np.array([0]), np.array([0]))
    vals = (np.array([0.92]), np.array([0.73666226]))
    res = [{'p0': vals[0][0] + vals[1][0]}]
    base_func(s, 2, ['conf', 'conf.p0', 'conf.p0a', 'conf.p0b'],
              ['gUniform', 'gGauss'], idxs, vals, res)


def test4():
    s = '{"p0" : uniform(0,1), "p1":this.p0}'
    idxs = (np.array([0]),)
    vals = ([np.array([0.72])],)
    res = [{"p0":vals[0][0][0], "p1":vals[0][0][0]}]
    base_func(s, 1, ['conf', 'conf.p0', 'conf.p1'], ['gUniform'],
              idxs, vals, res)


def test5():
    s = '{"p0" : uniform(0, 1), "p1":gaussian(this.p0, 1)}'
    idxs = (np.array([0]), np.array([0]))
    vals = (np.array([0.92]), np.array([0.73666226]))
    res = [{'p0': vals[0][0], 'p1': vals[1][0]}]
    base_func(s, 2, ['conf', 'conf.p0', 'conf.p1', 'conf.p1mean'],
              ['gUniform', 'gGauss'], idxs, vals, res)


def test6():
    s = '{"p0" : {"p1" : uniform(0, 1)}, "p2":gaussian(this.p0.p1, 1)}'
    idxs = (np.array([0]), np.array([0]))
    vals = (np.array([0.92]), np.array([0.73666226]))
    res = [{'p0': {'p1': 0.92}, 'p2': 0.73666226}]
    base_func(s, 2,
              ['conf', 'conf.p0', 'conf.p0.p1', 'conf.p2', 'conf.p2mean'],
              ['gUniform', 'gGauss'], idxs, vals, res)


def test7():
    s = """{"p0" : uniform(0, 1), "p1": gaussian(0, 1),
    "p2":choice([1, this.p0]), "p3":choice([2, this.p1])}"""
    idxs = (np.array([0]), np.array([0]), np.array([0]), np.array([0]))
    vals = (array([0.26]), array([0.32]), array([1], dtype=int32),
            array([1], dtype=int32))
    res = [{'p0': 0.26, 'p1': 0.32, 'p2': 0.26, 'p3':0.32}]
    base_func(s, 4, ['conf',
                   'conf.p0',
                   'conf.p1',
                   'conf.p2',
                   'conf.p2[0]',
                   'conf.p3',
                   'conf.p3[0]'],
              ['gUniform', 'gGauss', 'gChoice', 'gChoice'],
              idxs, vals, res)

def test_no_redundant_unions():
    s = """{"p0" : uniform(0, 1), "p1": gaussian(0, 1),
    "p2":choice([1, this.p0]), "p3":choice([2, this.p1])}"""
    template = gd.gDist(s)
    idxs, vals, s_N = template.theano_sampler(123)
    for v in theano.gof.graph.ancestors(idxs):
        assert v.owner is None or not isinstance(v.owner.op, gd.SetOp)

