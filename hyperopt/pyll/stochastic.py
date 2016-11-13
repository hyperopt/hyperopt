"""
Constructs for annotating base graphs.
"""
from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import sys
import numpy as np
from .base import scope, as_apply, dfs, rec_eval, clone

################################################################################
################################################################################


def ERR(msg):
    print(msg, file=sys.stderr)


implicit_stochastic_symbols = set()


def implicit_stochastic(f):
    implicit_stochastic_symbols.add(f.__name__)
    return f


@scope.define
def rng_from_seed(seed):
    return np.random.RandomState(seed)


# -- UNIFORM

@implicit_stochastic
@scope.define
def uniform(low, high, rng=None, size=()):
    return rng.uniform(low, high, size=size)


@implicit_stochastic
@scope.define
def loguniform(low, high, rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.exp(draw)


@implicit_stochastic
@scope.define
def quniform(low, high, q, rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.round(old_div(draw, q)) * q


@implicit_stochastic
@scope.define
def qloguniform(low, high, q, rng=None, size=()):
    draw = np.exp(rng.uniform(low, high, size=size))
    return np.round(old_div(draw, q)) * q


# -- NORMAL

@implicit_stochastic
@scope.define
def normal(mu, sigma, rng=None, size=()):
    return rng.normal(mu, sigma, size=size)


@implicit_stochastic
@scope.define
def qnormal(mu, sigma, q, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.round(old_div(draw, q)) * q


@implicit_stochastic
@scope.define
def lognormal(mu, sigma, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.exp(draw)


@implicit_stochastic
@scope.define
def qlognormal(mu, sigma, q, rng=None, size=()):
    draw = np.exp(rng.normal(mu, sigma, size=size))
    return np.round(old_div(draw, q)) * q


# -- CATEGORICAL


@implicit_stochastic
@scope.define
def randint(upper, rng=None, size=()):
    # this is tricky because numpy doesn't support
    # upper being a list of len size[0]
    if isinstance(upper, (list, tuple)):
        if isinstance(size, int):
            assert len(upper) == size
            return np.asarray([rng.randint(uu) for uu in upper])
        elif len(size) == 1:
            assert len(upper) == size[0]
            return np.asarray([rng.randint(uu) for uu in upper])
    return rng.randint(upper, size=size)


@implicit_stochastic
@scope.define
def categorical(p, upper=None, rng=None, size=()):
    """Draws i with probability p[i]"""
    if len(p) == 1 and isinstance(p[0], np.ndarray):
        p = p[0]
    p = np.asarray(p)

    if size == ():
        size = (1,)
    elif isinstance(size, (int, np.number)):
        size = (size,)
    else:
        size = tuple(size)

    if size == (0,):
        return np.asarray([])
    assert len(size)

    if p.ndim == 0:
        raise NotImplementedError()
    elif p.ndim == 1:
        n_draws = int(np.prod(size))
        sample = rng.multinomial(n=1, pvals=p, size=int(n_draws))
        assert sample.shape == size + (len(p),)
        rval = np.dot(sample, np.arange(len(p)))
        rval.shape = size
        return rval
    elif p.ndim == 2:
        n_draws_, n_choices = p.shape
        n_draws, = size
        assert n_draws == n_draws_
        rval = [np.where(rng.multinomial(pvals=p[ii], n=1))[0][0]
                for ii in range(n_draws)]
        rval = np.asarray(rval)
        rval.shape = size
        return rval
    else:
        raise NotImplementedError()


def choice(args):
    return scope.one_of(*args)
scope.choice = choice


def one_of(*args):
    ii = scope.randint(len(args))
    return scope.switch(ii, *args)
scope.one_of = one_of


def recursive_set_rng_kwarg(expr, rng=None):
    """
    Make all of the stochastic nodes in expr use the rng

    uniform(0, 1) -> uniform(0, 1, rng=rng)
    """
    if rng is None:
        rng = np.random.RandomState()
    lrng = as_apply(rng)
    for node in dfs(expr):
        if node.name in implicit_stochastic_symbols:
            for ii, (name, arg) in enumerate(list(node.named_args)):
                if name == 'rng':
                    node.named_args[ii] = ('rng', lrng)
                    break
            else:
                node.named_args.append(('rng', lrng))
    return expr


def sample(expr, rng=None, **kwargs):
    """
    Parameters:
    expr - a pyll expression to be evaluated

    rng - a np.random.RandomState instance
          default: `np.random.RandomState()`

    **kwargs - optional arguments passed along to
               `hyperopt.pyll.rec_eval`

    """
    if rng is None:
        rng = np.random.RandomState()
    foo = recursive_set_rng_kwarg(clone(as_apply(expr)), as_apply(rng))
    return rec_eval(foo, **kwargs)
