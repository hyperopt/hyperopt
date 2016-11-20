from __future__ import print_function
from builtins import map
from hyperopt.pyll_utils import EQ
from hyperopt.pyll_utils import expr_to_config
from hyperopt import hp
from hyperopt.pyll import as_apply


def test_expr_to_config():

    z = hp.randint('z', 10)
    a = hp.choice('a',
                  [
                      hp.uniform('b', -1, 1) + z,
                      {'c': 1, 'd': hp.choice('d',
                                              [3 + hp.loguniform('c', 0, 1),
                                               1 + hp.loguniform('e', 0, 1)])
                       }])

    expr = as_apply((a, z))

    hps = {}
    expr_to_config(expr, (True,), hps)

    for label, dct in list(hps.items()):
        print(label)
        print('  dist: %s(%s)' % (
            dct['node'].name,
            ', '.join(map(str, [ii.eval() for ii in dct['node'].inputs()]))))
        if len(dct['conditions']) > 1:
            print('  conditions (OR):')
            for condseq in dct['conditions']:
                print('    ', ' AND '.join(map(str, condseq)))
        elif dct['conditions']:
            for condseq in dct['conditions']:
                print('  conditions :', ' AND '.join(map(str, condseq)))

    assert hps['a']['node'].name == 'randint'
    assert hps['b']['node'].name == 'uniform'
    assert hps['c']['node'].name == 'loguniform'
    assert hps['d']['node'].name == 'randint'
    assert hps['e']['node'].name == 'loguniform'
    assert hps['z']['node'].name == 'randint'

    assert set([(True, EQ('a', 0))]) == set([(True, EQ('a', 0))])
    assert hps['a']['conditions'] == set([(True,)])
    assert hps['b']['conditions'] == set([
        (True, EQ('a', 0))]), hps['b']['conditions']
    assert hps['c']['conditions'] == set([
        (True, EQ('a', 1), EQ('d', 0))])
    assert hps['d']['conditions'] == set([
        (True, EQ('a', 1))])
    assert hps['e']['conditions'] == set([
        (True, EQ('a', 1), EQ('d', 1))])
    assert hps['z']['conditions'] == set([
        (True,),
        (True, EQ('a', 0))])


def test_remove_allpaths():
    z = hp.uniform('z', 0, 10)
    a = hp.choice('a', [z + 1, z - 1])
    hps = {}
    expr_to_config(a, (True,), hps)
    aconds = hps['a']['conditions']
    zconds = hps['z']['conditions']
    assert aconds == set([(True,)]), aconds
    assert zconds == set([(True,)]), zconds
