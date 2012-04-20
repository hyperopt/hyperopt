import pyll
from pyll import scope


@scope.define
def hyperopt_param(label, obj):
    """ A graph node primarily for annotating - VectorizeHelper looks out
    for these guys, and optimizes subgraphs of the form:

        hyperopt_param(<stochastic_expression>(...))

    """
    return obj


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

