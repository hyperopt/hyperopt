import pyll


@pyll.scope.define
def hyperopt_param(label, obj):
    """ A graph node primarily for annotating - VectorizeHelper looks out
    for these guys, and optimizes subgraphs of the form:

        hyperopt_param(<stochastic_expression>(...))

    """
    return obj


def hp_choice(label, options):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    ch = pyll.scope.hyperopt_param(label,
        pyll.scope.randint(len(options)))
    return pyll.scope.switch(ch, *options)


def hp_randint(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
        pyll.scope.randint(*args, **kwargs))


def hp_uniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.uniform(*args, **kwargs))


def hp_quniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.quniform(*args, **kwargs))


def hp_loguniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.loguniform(*args, **kwargs))


def hp_qloguniform(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.qloguniform(*args, **kwargs))


def hp_normal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.normal(*args, **kwargs))


def hp_qnormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.qnormal(*args, **kwargs))


def hp_lognormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.lognormal(*args, **kwargs))


def hp_qlognormal(label, *args, **kwargs):
    if not isinstance(label, basestring):
        raise TypeError('require string label')
    return pyll.scope.hyperopt_param(label,
            pyll.scope.qlognormal(*args, **kwargs))

