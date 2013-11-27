from functools import partial
from base import DuplicateLabel
from pyll.base import Apply
from pyll import scope
from pyll import as_apply

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


def hp_pchoice(label, p_options):
    """
    label: string
    p_options: list of (probability, option) pairs
    """
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    p, options = zip(*p_options)
    n_options = len(options)
    ch = scope.hyperopt_param(label,
                              scope.categorical(
                                  p,
                                  upper=n_options))
    return scope.switch(ch, *options)


def hp_choice(label, options):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    ch = scope.hyperopt_param(label,
        scope.randint(len(options)))
    return scope.switch(ch, *options)


def hp_randint(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.hyperopt_param(label,
        scope.randint(*args, **kwargs))


def hp_uniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.uniform(*args, **kwargs)))


def hp_quniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.quniform(*args, **kwargs)))


def hp_loguniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.loguniform(*args, **kwargs)))


def hp_qloguniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.qloguniform(*args, **kwargs)))


def hp_normal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.normal(*args, **kwargs)))


def hp_qnormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.qnormal(*args, **kwargs)))


def hp_lognormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return scope.float(
            scope.hyperopt_param(label,
                scope.lognormal(*args, **kwargs)))


def hp_qlognormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
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
        return 'Cond{%s %s %s}' %  (self.name, self.op, self.val)

    def __eq__(self, other):
        return self.op == other.op and self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.op, self.name, self.val))

    def __repr__(self):
        return str(self)

EQ = partial(Cond, op='=')

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
    if expr.name == 'switch':
        idx = expr.inputs()[0]
        options = expr.inputs()[1:]
        assert idx.name == 'hyperopt_param'
        assert idx.arg['obj'].name in (
                'randint',     # -- in case of hp.choice
                'categorical', # -- in case of hp.pchoice
                )
        expr_to_config(idx, conditions, hps)
        for ii, opt in enumerate(options):
            expr_to_config(opt,
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
            expr_to_config(ii, conditions, hps)

