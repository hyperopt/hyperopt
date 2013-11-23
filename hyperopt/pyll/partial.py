"""
Support code for `functools.partial` based deferred-evaluation
mechanism.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

import compiler
import functools


class NoExpand(object):
    def __init__(self, node):
        self.node = node


_BINARY_OPS = {'+': lambda x, y: x + y,
               '-': lambda x, y: x - y,
               '*': lambda x, y: x * y,
               '/': lambda x, y: x / y,
               '%': lambda x, y: x % y,
               '^': lambda x, y: x ^ y,
               '&': lambda x, y: x & y,
               '|': lambda x, y: x | y,
               '**': lambda x, y: x ** y,
               '//': lambda x, y: x // y,
               '==': lambda x, y: x == y,
               '!=': lambda x, y: x != y,
               '>': lambda x, y: x > y,
               '<': lambda x, y: x < y,
               '>=': lambda x, y: x >= y,
               '<=': lambda x, y: x <= y,
               '<<': lambda x, y: x << y,
               '>>': lambda x, y: x >> y,
               'and': lambda x, y: x and y,
               'or': lambda x, y: x or y}

_UNARY_OPS = {'+': lambda x: +x,
              '-': lambda x: -x,
              '~': lambda x: ~x}


def _binary_arithmetic(x, y, op):
    return _BINARY_OPS[op](x, y)


def _unary_arithmetic(x, op):
    return _UNARY_OPS[op](x)


def _getitem(obj, item):
    return obj[item]


class MissingArgument(object):
    """Object to represent a missing argument to a function application
    """
    def __init__(self):
        assert 0, "Singleton class not meant to be instantiated"


def _param_assignment(pp):
    """Calculate parameter assignment of partial
    """
    binding = {}

    fn = pp.func
    pos_args = pp.args
    named_args = {} if pp.keywords is None else pp.keywords
    code = fn.__code__
    extra_args_ok = bool(code.co_flags & compiler.consts.CO_VARARGS)
    extra_kwargs_ok = bool(code.co_flags & compiler.consts.CO_VARKEYWORDS)

    if extra_args_ok and extra_kwargs_ok:
        assert len(code.co_varnames) >= code.co_argcount + 2
        param_names = code.co_varnames[:code.co_argcount + 2]
        args_param = param_names[code.co_argcount]
        kwargs_param = param_names[code.co_argcount + 1]
        pos_params = param_names[:code.co_argcount]
    elif extra_kwargs_ok:
        assert len(code.co_varnames) >= code.co_argcount + 1
        param_names = code.co_varnames[:code.co_argcount + 1]
        kwargs_param = param_names[code.co_argcount]
        pos_params = param_names[:code.co_argcount]
    elif extra_args_ok:
        assert len(code.co_varnames) >= code.co_argcount + 1
        param_names = code.co_varnames[:code.co_argcount + 1]
        args_param = param_names[code.co_argcount]
        pos_params = param_names[:code.co_argcount]
    else:
        assert len(code.co_varnames) >= code.co_argcount
        param_names = code.co_varnames[:code.co_argcount]
        pos_params = param_names[:code.co_argcount]

    if extra_args_ok:
        binding[args_param] = []

    if extra_kwargs_ok:
        binding[kwargs_param] = {}

    if len(pos_args) > code.co_argcount and not extra_args_ok:
        raise TypeError('Argument count exceeds number of positional params')

    # -- bind positional arguments
    for param_i, arg_i in zip(pos_params, pos_args):
        binding[param_i] = arg_i

    if extra_args_ok:
        binding[args_param].extend(pos_args[code.co_argcount:])

    # -- bind keyword arguments
    for aname, aval in named_args.items():
        try:
            pos = pos_params.index(aname)
        except ValueError:
            if extra_kwargs_ok:
                binding[kwargs_param][aname] = aval
                continue
            else:
                raise TypeError('Unrecognized keyword argument', aname)
        param = param_names[pos]
        if param in binding:
            raise TypeError('Duplicate argument for parameter', param)
        binding[param] = aval

    assert len(binding) <= len(param_names)

    # -- fill in default parameter values
    if defaults:
        for param_i, default_i in zip(pos_params[-len(defaults)], defaults):
            binding.setdefault(param_i, default_i)

    # -- mark any outstanding parameters as missing
    if len(binding) < len(param_names):
        for p in param_names:
            if p not in binding:
                binding[p] = MissingArgument

    return binding


class PartialPlus(functools.partial):
    """
    A subclass of `functools.partial` that allows for
    common arithmetic/builtin operations to be performed
    on them, deferred by wrapping in another object of
    this same type. Also overrides `__call__` to suggest
    you use the recursive version, `evaluate`.

    Notable exceptions *not* implemented include __len__ and
    __iter__, because returning non-integer/iterator stuff
    from those methods tends to break things.
    """

    def __call__(self, *args, **kwargs):
        raise TypeError("use evaluate() for %s objects" %
                        self.__class__.__name__)

    def __add__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '+')

    def __sub__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '-')

    def __mul__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '*')

    def __floordiv__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '//')

    def __mod__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '%')

    def __divmod__(self, other):
        return self.__class__(divmod, self, other)

    def __pow__(self, other, modulo=None):
        return self.__class__(pow, self, other, modulo)

    def __lshift__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '<<')

    def __rshift__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '>>')

    def __and__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '&')

    def __xor__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '^')

    def __or__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '|')

    def __div__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '/')

    def __truediv__(self, other):
        return self.__class__(_binary_arithmetic, self, other, '/')

    def __lt__(self, other):
        return _binary_arithmetic(self, other, '<')

    def __le__(self, other):
        return _binary_arithmetic(self, other, '<=')

    def __eq__(self, other):
        return _binary_arithmetic(self, other, '==')

    def __ne__(self, other):
        return _binary_arithmetic(self, other, '!=')

    def __gt__(self, other):
        return _binary_arithmetic(self, other, '>')

    def __ge__(self, other):
        return _binary_arithmetic(self, other, '>=')

    def __neg__(self):
        return self.__class__(_unary_arithmetic, self, '-')

    def __pos__(self):
        return self.__class__(_unary_arithmetic, self, '+')

    def __abs__(self):
        return self.__class__(abs, self)

    def __invert__(self):
        return self.__class__(abs, self)

    def __complex__(self):
        return self.__class__(complex, self)

    def __int__(self):
        return self.__class__(int, self)

    def __long__(self):
        return self.__class__(long, self)

    def __float__(self):
        return self.__class__(float, self)

    def __oct__(self):
        return self.__class__(oct, self)

    def __hex__(self):
        return self.__class__(hex, self)

    def __getitem__(self, item):
        return self.__class__(_getitem, self, item)

    @property
    def arg(self):
        return _param_assignment(self)


def evaluate(p, cache=None):
    """
    Evaluate a nested tree of functools.partial objects,
    used for deferred evaluation.

    Parameters
    ----------
    p : object
        If `p` is a partial, or a subclass of partial, it is
        expanded recursively. Otherwise th

    Returns
    -------
    q : object
        The result of evaluating `p` if `p` was a partial
        instance, or else `p` itself.

    Notes
    -----
    For large graphs this recursive implementation may hit the
    recursion limit and be kind of slow. TODO: write an
    iterative version.
    """
    cache = {} if cache is None else cache
    if isinstance(p, NoExpand):
        return p.node
    if not isinstance(p, functools.partial):
        return p
    # If we've encountered this exact partial node before,
    # short-circuit the evaluation of this branch and return
    # the pre-computed value.
    if p in cache:
        return cache[p]

    # When evaluating an expression of the form
    # `list(...)[item]`
    # only evaluate the element(s) of the list that we need.
    if p.func == _getitem:
        obj, item = p.args
        if (isinstance(obj, functools.partial)
                and obj.func in (list, tuple)):
            item_val = evaluate(item, cache)
            elem_val = evaluate(obj.args[item_val], cache)
            try:
                int(item_val)
                cache[p] = elem_val
            except TypeError:
                cache[p] = obj.func(elem_val)
            return cache[p]

    args = [evaluate(arg, cache) for arg in p.args]
    if p.keywords:
        kw = [evaluate(kw, cache) for kw in p.keywords]
    else:
        kw = {}
    # Cache the evaluated value (for subsequent calls that
    # will look at this cache dictionary) and return.
    cache[p] = p.func(*args, **kw)
    return cache[p]
