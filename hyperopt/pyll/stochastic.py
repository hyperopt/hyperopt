"""
Constructs for annotating base graphs.
"""
import sys
import numpy as np

from .base import scope, as_apply, dfs, Apply, rec_eval, clone

################################################################################
################################################################################
def ERR(msg):
    print >> sys.stderr, msg


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
    return np.round(draw/q) * q


@implicit_stochastic
@scope.define
def qloguniform(low, high, q, rng=None, size=()):
    draw = np.exp(rng.uniform(low, high, size=size))
    return np.round(draw/q) * q


# -- NORMAL

@implicit_stochastic
@scope.define
def normal(mu, sigma, rng=None, size=()):
    return rng.normal(mu, sigma, size=size)


@implicit_stochastic
@scope.define
def qnormal(mu, sigma, q, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.round(draw/q) * q


@implicit_stochastic
@scope.define
def lognormal(mu, sigma, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.exp(draw)


@implicit_stochastic
@scope.define
def qlognormal(mu, sigma, q, rng=None, size=()):
    draw = np.exp(rng.normal(mu, sigma, size=size))
    return np.round(draw/q) * q


# -- CATEGORICAL


@implicit_stochastic
@scope.define
def randint(upper, rng=None, size=()):
    # this is tricky because numpy doesn't support
    # upper being a list of len size[0]
    asdf = 9
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
def categorical(p, rng=None, size=()):
    """Draws i with probability p[i]"""
    #XXX: OMG this is the craziest shit
    p = np.asarray(p)
    if isinstance(size, (int, np.number)):
        size = (size,)
    else:
        size = tuple(size)
    n_draws = np.prod(size)
    sample = rng.multinomial(n=1, pvals=p, size=size)
    assert sample.shape == size + (len(p),)
    if size:
        rval = np.sum(sample * np.arange(len(p)), axis=len(size))
    else:
        rval = [np.where(rng.multinomial(pvals=p, n=1))[0][0]
                for i in xrange(n_draws)]
        rval = np.asarray(rval, dtype=self.otype.dtype)
    rval.shape = size
    return rval


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
            node.named_args.append(('rng', lrng))
    return expr


def sample(expr, rng=None, **kwargs):
    if rng is None:
        rng = np.random.RandomState()
    foo = recursive_set_rng_kwarg(clone(as_apply(expr)), as_apply(rng))
    return rec_eval(foo, **kwargs)
