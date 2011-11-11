import cPickle
import hyperopt.gdist as gd
import theano
import numpy as np
from numpy import array, int32
import genson
import collections
from bson import SON


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


class WrongLen(Exception): pass
class WrongRandomLen(Exception): pass

def pickle_helper_len(s):
    parser = genson.parser.GENSONParser()
    genson_obj = parser.parse_string(s)
    genson_obj2 = cPickle.loads(cPickle.dumps(genson_obj))
    assert genson_obj == genson_obj2

    def flatten(t, seen=None):
        """Return all children recursively as list.
        """
        if seen is None:
            seen = set()
        if isinstance(t, collections.OrderedDict):
            return [(k, flatten(v, seen)) for k, v in t.items()]
        elif isinstance(t, list):
            return [flatten(v, seen) for v in t]
        else:
            raise TypeError(t, type(t))

    f1 = flatten(genson_obj)
    f2 = flatten(genson_obj2)
    assert len(f1) == len(f2)

    template = gd.gDict(genson_obj)
    template2 = cPickle.loads(cPickle.dumps([template]))[0]
    t1 = template.flatten()
    t2 = template2.flatten()
    print [(k, type(v)) for k,v in template.items()]
    print [(k, type(v)) for k,v in template2.items()]
    assert len(t1) == len(t2)


def pickle_helper(s):
    #pickle_helper_len(s)

    template = gd.gDist(s)

    template2 = cPickle.loads(cPickle.dumps(template, protocol=1))

    assert template.sample(5) == template2.sample(5)

    idxs, vals, s_N = template.theano_sampler(5)
    idxs2, vals2, s_N2 = template.theano_sampler(5)
    len(idxs) == len(idxs2)
    len(vals) == len(vals2)

    if len(template.flatten()) != len(template2.flatten()):
        tf1 = template.flatten()
        tf2 = template2.flatten()
        for i in range(max(len(tf1), len(tf1))):
            print '----'
            print i
            if i < len(tf1):
                print tf1[i]
            else:
                print 'N/A'
            if i < len(tf2):
                print tf2[i]
            else:
                print 'N/A'
        raise WrongLen()

    if len(template.random_nodes()) != len(template2.random_nodes()):
        raise WrongRandomLen()

    assert template == template2

def test_SON():
    a = SON([])
    b = SON([('a', a), ('alist', [a, a])])

    def foo(p):
        try:
            bb = cPickle.loads(cPickle.dumps(b, protocol=p))
            assert bb['a'] is bb['alist'][0]
            assert bb['a'] is bb['alist'][1]
        except:
            print 'fail in protocol level', p
            return 0
        return 1

    passes = [foo(i) for i in [0, 1, 2, -1]]
    assert all(passes)


def test_pickle():
    pickle_helper("""{"p0" : uniform(0, 1), "p1": gaussian(0, 1),
    "p2":choice([1, this.p0]), "p3":choice([2, this.p1])}""")

def test_pickle_corner_case():
    # -- triggered pickling Nov 10
    pickle_helper('{ "n_filters": [], "generate": [[]] }')

def test_pickle_list():
    pickle_helper("""{"p0" : []}""")
    pickle_helper("""{"p0" : [0]}""")
    pickle_helper("""{"p0" : [0, 1, 2, 3, 4]}""")
    pickle_helper("""{"p0" : [uniform(0, 1), 0]}""")
    pickle_helper("""{"p0" : [uniform(0, 1), gaussian(0, 1)]}""")
    pickle_helper("""{"p0" : [uniform(0, 1), choice([0, 1])]}""")
    pickle_helper("""{"p0" : [uniform(0, 1), choice([0, 1, gaussian(0, 1)])]}""")


def test_pickle_lnorm():
    s = """{"lnorm": ("lnorm", {"kwargs":
    {"inker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "outker_shape":
    this.inker_shape, "stretch": lognormal(0, 1), "threshold": lognormal(0, 1),
    "remove_mean": choice([0, 1])}})}"""
    pickle_helper(s)

def test_pickle_fbcorr():
    s = """{"fbcorr": (
    "fbcorr",
    { "initialize": {
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
        "generate": ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})
        },
      "kwargs": {
        "max_out": choice([1, null]),
        "min_out": choice([null, 0])}}
    )}"""
    pickle_helper(s)

def test_pickle_fbcorr2():
    s = """{"fbcorr": (
    "fbcorr",
    { "initialize": {
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
        "generate": ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})
        }})}"""
    pickle_helper(s)

def test_pickle_fbcorr3():
    s = """{"fbcorr":
    { "initialize": {
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
        "generate": ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})
        }}}"""
    pickle_helper(s)

def test_pickle_fbcorr4():
    s = """{"fbcorr":
    {
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]),
        "generate": ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})
        }}"""
    pickle_helper(s)

def test_pickle_fbcorr5():
    s = """{"fbcorr": {
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "generate": ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})
        }}"""
    pickle_helper(s)

def test_pickle_fbcorr6():
    s = """{
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "generate": ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})
        }"""
    pickle_helper(s)

def test_pickle_fbcorr7():
    s = """{
        "n_filters": qlognormal(3.4657359027997265, 1, round=16),
        "generate": ("random:uniform", {"rseed": 0})
        }"""
    pickle_helper(s)

def test_pickle_fbcorr8():
    s = """{
        "n_filters": uniform(0, 1),
        "generate": (0, {"rseed": 0})
        }"""
    pickle_helper(s)

def test_pickle_fbcorr9():
    s = """{
        "n_filters": [],
        "generate": [[]]
        }"""
    pickle_helper(s)


def test_pickle_lpool():
    s = """{"lpool":("lpool",
    {"kwargs": {"ker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "order":
    choice([1, 2, 10, uniform(1, 10)]), "stride": 2}})}
    """
    pickle_helper(s)

def test_pickle_layer():
    s = """{"layer": [("fbcorr",
    {"initialize": {"n_filters": qlognormal(3.4657359027997265, 1, round=16),
    "filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "generate":
    ("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})}, "kwargs":
    {"max_out": choice([1, null]), "min_out": choice([null, 0])}}), ("lpool",
    {"kwargs": {"ker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "order":
    choice([1, 2, 10, uniform(1, 10)]), "stride": 2}}), ("lnorm", {"kwargs":
    {"inker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "outker_shape":
    this.inker_shape, "stretch": lognormal(0, 1), "threshold": lognormal(0, 1),
    "remove_mean": choice([0, 1])}})]}
    """
    pickle_helper(s)

def test_pickle_big():
    pickle_helper(big_test_string)


big_test_string = """
{"comparison": ["mult", "absdiff", "sqrtabsdiff", "sqdiff"], "desc":
[[("lnorm", {"kwargs": {"inker_shape": choice([(3, 3), (5, 5), (7, 7), (9,
9)]), "outker_shape": this.inker_shape, "stretch": lognormal(0, 1),
"threshold": lognormal(0, 1), "remove_mean": choice([0, 1])}})], [("fbcorr",
{"initialize": {"n_filters": qlognormal(3.4657359027997265, 1, round=16),
"filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "generate":
("random:uniform", {"rseed": choice([11, 12, 13, 14, 15])})}, "kwargs":
{"max_out": choice([1, null]), "min_out": choice([null, 0])}}), ("lpool",
{"kwargs": {"ker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "order":
choice([1, 2, 10, uniform(1, 10)]), "stride": 2}}), ("lnorm", {"kwargs":
{"inker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "outker_shape":
this.inker_shape, "stretch": lognormal(0, 1), "threshold": lognormal(0, 1),
"remove_mean": choice([0, 1])}})], [("fbcorr", {"initialize": {"n_filters":
qlognormal(3.4657359027997265, 1, round=16), "filter_shape": choice([(3, 3),
(5, 5), (7, 7), (9, 9)]), "generate": ("random:uniform", {"rseed": choice([21,
22, 23, 24, 25])})}, "kwargs": {"max_out": choice([1, null]), "min_out":
choice([null, 0])}}), ("lpool", {"kwargs": {"ker_shape": choice([(3, 3), (5,
5), (7, 7), (9, 9)]), "order": choice([1, 2, 10, uniform(1, 10)]), "stride":
2}}), ("lnorm", {"kwargs": {"inker_shape": choice([(3, 3), (5, 5), (7, 7), (9,
9)]), "outker_shape": this.inker_shape, "stretch": lognormal(0, 1),
"threshold": lognormal(0, 1), "remove_mean": choice([0, 1])}})], [("fbcorr",
{"initialize": {"n_filters": qlognormal(3.4657359027997265, 1, round=16),
"filter_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "generate":
("random:uniform", {"rseed": choice([31, 32, 33, 34, 35])})}, "kwargs":
{"max_out": choice([1, null]), "min_out": choice([null, 0])}}), ("lpool",
{"kwargs": {"ker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "order":
choice([1, 2, 10, uniform(1, 10)]), "stride": 2}}), ("lnorm", {"kwargs":
{"inker_shape": choice([(3, 3), (5, 5), (7, 7), (9, 9)]), "outker_shape":
this.inker_shape, "stretch": lognormal(0, 1), "threshold": lognormal(0, 1),
"remove_mean": choice([0, 1])}})]]}
"""

