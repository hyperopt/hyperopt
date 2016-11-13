from __future__ import absolute_import
from builtins import str
from builtins import zip
from builtins import range
from past.builtins import basestring
from builtins import object
from functools import partial, wraps
from .base import DuplicateLabel
from .pyll.base import Apply, Literal
from .pyll import scope
from .pyll import as_apply


def validate_label(f):
    @wraps(f)
    def wrapper(label, *args, **kwargs):
        is_real_string = isinstance(label, basestring)
        is_literal_string = (isinstance(label, Literal) and
                             isinstance(label.obj, basestring))
        if not is_real_string and not is_literal_string:
            raise TypeError('require string label')
        return f(label, *args, **kwargs)
    return wrapper


#
# Hyperparameter Types
#

@scope.define
def hyperopt_param(label, obj):
    """ A graph node primarily for annotating - VectorizeHelper looks out
    for these guys, and optimizes subgraphs of the form:

        hyperopt_param(<stochastic_expression>(...))

    """
    return obj


@validate_label
def hp_pchoice(label, p_options):
    """
    label: string
    p_options: list of (probability, option) pairs
    """
    p, options = list(zip(*p_options))
    n_options = len(options)
    ch = scope.hyperopt_param(label,
                              scope.categorical(
                                  p,
                                  upper=n_options))
    return scope.switch(ch, *options)


@validate_label
def hp_choice(label, options):
    ch = scope.hyperopt_param(label,
                              scope.randint(len(options)))
    return scope.switch(ch, *options)


@validate_label
def hp_randint(label, *args, **kwargs):
    return scope.hyperopt_param(label,
                                scope.randint(*args, **kwargs))


@validate_label
def hp_uniform(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.uniform(*args, **kwargs)))


@validate_label
def hp_quniform(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.quniform(*args, **kwargs)))


@validate_label
def hp_loguniform(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.loguniform(*args, **kwargs)))


@validate_label
def hp_qloguniform(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.qloguniform(*args, **kwargs)))


@validate_label
def hp_normal(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.normal(*args, **kwargs)))


@validate_label
def hp_qnormal(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.qnormal(*args, **kwargs)))


@validate_label
def hp_lognormal(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.lognormal(*args, **kwargs)))


@validate_label
def hp_qlognormal(label, *args, **kwargs):
    return scope.float(
        scope.hyperopt_param(label,
                             scope.qlognormal(*args, **kwargs)))


#
# Tools for extracting a search space from a Pyll graph
#


class Cond(object):

    def __init__(self, name, val, op):
        self.op = op
        self.name = name
        self.val = val

    def __str__(self):
        return 'Cond{%s %s %s}' % (self.name, self.op, self.val)

    def __eq__(self, other):
        return self.op == other.op and self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.op, self.name, self.val))

    def __repr__(self):
        return str(self)

EQ = partial(Cond, op='=')


def _expr_to_config(expr, conditions, hps):
    if expr.name == 'switch':
        idx = expr.inputs()[0]
        options = expr.inputs()[1:]
        assert idx.name == 'hyperopt_param'
        assert idx.arg['obj'].name in (
            'randint',     # -- in case of hp.choice
            'categorical',  # -- in case of hp.pchoice
        )
        _expr_to_config(idx, conditions, hps)
        for ii, opt in enumerate(options):
            _expr_to_config(opt,
                            conditions + (EQ(idx.arg['label'].obj, ii),),
                            hps)
    elif expr.name == 'hyperopt_param':
        label = expr.arg['label'].obj
        if label in hps:
            if hps[label]['node'] != expr.arg['obj']:
                raise DuplicateLabel(label)
            hps[label]['conditions'].add(conditions)
        else:
            hps[label] = {'node': expr.arg['obj'],
                          'conditions': set((conditions,)),
                          'label': label,
                          }
    else:
        for ii in expr.inputs():
            _expr_to_config(ii, conditions, hps)


def expr_to_config(expr, conditions, hps):
    """
    Populate dictionary `hps` with the hyperparameters in pyll graph `expr`
    and conditions for participation in the evaluation of `expr`.

    Arguments:
    expr       - a pyll expression root.
    conditions - a tuple of conditions (`Cond`) that must be True for
                 `expr` to be evaluated.
    hps        - dictionary to populate

    Creates `hps` dictionary:
        label -> { 'node': apply node of hyperparameter distribution,
                   'conditions': `conditions` + tuple,
                   'label': label
                   }
    """
    expr = as_apply(expr)
    if conditions is None:
        conditions = ()
    assert isinstance(expr, Apply)
    _expr_to_config(expr, conditions, hps)
    _remove_allpaths(hps, conditions)


def _remove_allpaths(hps, conditions):
    """Hacky way to recognize some kinds of false dependencies
    Better would be logic programming.
    """
    potential_conds = {}
    for k, v in list(hps.items()):
        if v['node'].name in ('randint', 'categorical'):
            upper = v['node'].arg['upper'].obj
            potential_conds[k] = frozenset([EQ(k, ii) for ii in range(upper)])

    for k, v in list(hps.items()):
        if len(v['conditions']) > 1:
            all_conds = [[c for c in cond if c is not True]
                         for cond in v['conditions']]
            all_conds = [cond for cond in all_conds if len(cond) >= 1]
            if len(all_conds) == 0:
                v['conditions'] = set([conditions])
                continue

            depvar = all_conds[0][0].name

            all_one_var = all(len(cond) == 1 and cond[0].name == depvar
                              for cond in all_conds)
            if all_one_var:
                conds = [cond[0] for cond in all_conds]
                if frozenset(conds) == potential_conds[depvar]:
                    v['conditions'] = set([conditions])
                    continue


# -- eof
