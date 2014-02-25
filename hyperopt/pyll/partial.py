"""
Support code for `functools.partial` based deferred-evaluation
mechanism.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

import compiler
import functools
import warnings
from itertools import izip, repeat


def switch(index, *args):
    """
    A switch statement treated specially by `evaluate`.

    Parameters
    ----------
    index : int
        Which argument branch to evaluate.
    *args : object, must have at least 1
        The arguments from which to select.

    Returns
    -------
    branch : object
        Object corresponding to `args[index]`.
    """
    return args[index]


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


def _extract_param_names(fn):
    """
    Grab the names of positional arguments, as well as the varargs
    and kwargs parameter, if they exist.

    Parameters
    ----------
    fn : function object
        The function to be inspected.

    Returns
    -------
    param_names : list
        A list of all the function's argument names.

    pos_args : list
        A list of names of the non-special arguments to `fn`.

    args_param : str or None
        The name of the variable-length positional args parameter,
        or `None` if `fn` does not accept a variable number of
        positional arguments.

    kwargs_param : str or None
        The name of the variable-length keyword args parameter,
        or `None` if `fn` does not accept a variable number of
        keyword arguments.
    """
    code = fn.__code__

    extra_args_ok = bool(code.co_flags & compiler.consts.CO_VARARGS)
    extra_kwargs_ok = bool(code.co_flags & compiler.consts.CO_VARKEYWORDS)
    expected_num_args = (code.co_argcount + int(extra_args_ok) +
                         int(extra_kwargs_ok))
    assert len(code.co_varnames) >= expected_num_args
    param_names = code.co_varnames[:expected_num_args]
    args_param = (param_names[code.co_argcount]
                  if extra_args_ok else None)
    kwargs_param = (param_names[code.co_argcount + int(extra_args_ok)]
                    if extra_kwargs_ok else None)
    pos_params = param_names[:code.co_argcount]
    return pos_params, args_param, kwargs_param


def _bind_parameters(params, named_args, kwargs_param, binding=None):
    """
    Resolve bindings for arguments from a list of parameter
    names.

    Parameters
    ----------
    params : list
        A list of names of positional parameters.

    named_args : dict
        A dictionary mapping names of keyword parameters to
        values to bind to them.

    kwargs_param : str or None
        The name of the extended/optional keywords parameter
        to use for keys in `named_args` that do not appear in
        `params`. If this is None, excess keyword arguments
        not listed in `params` will raise an error.

    binding : dict, optional
        A dictionary of existing name to value bindings, i.e.
        from processing positional arguments.

    Returns
    -------
    binding : dict
        A dictionary of argument names to bound values, including
        any passed in via the `binding` argument.
    """
    binding = {} if binding is None else dict(binding)
    if kwargs_param:
        binding[kwargs_param] = {}
    params_set = set(params)
    for aname, aval in named_args.iteritems():
        if aname in params_set and not aname in binding:
            binding[aname] = aval
        elif aname in binding and aname != kwargs_param:
            raise TypeError('Duplicate argument for parameter: %s' % aname)
        elif kwargs_param:
            binding[kwargs_param][aname] = aval
        else:
            raise TypeError('Unrecognized keyword argument: %s' % aname)
    return binding


def _param_assignment(pp):
    """
    Calculate parameter assignment of partial
    """
    binding = {}

    fn = pp.func
    code = fn.__code__
    pos_args = pp.args
    named_args = {} if pp.keywords is None else pp.keywords
    params, args_param, kwargs_param = _extract_param_names(fn)

    if len(pos_args) > code.co_argcount and not args_param:
        raise TypeError('Argument count exceeds number of positional params')
    elif args_param:
        binding[args_param] = pos_args[code.co_argcount:]

    # -- bind positional arguments
    for param_i, arg_i in izip(params, pos_args):
        binding[param_i] = arg_i

    # -- bind keyword arguments
    binding.update(_bind_parameters(params, named_args, kwargs_param, binding))
    expected_length = (len(params) + int(kwargs_param is not None) +
                       int(args_param is not None))
    assert len(binding) <= expected_length

    # Right-aligned default values for params. Default to empty tuple
    # so that iteration below simply terminates in this case.
    defaults = fn.__defaults__ if fn.__defaults__ else ()

    # -- fill in default parameter values
    for param_i, default_i in izip(params[-len(defaults):], defaults):
        binding.setdefault(param_i, default_i)

    # -- mark any outstanding parameters as missing
    missing_names = set(params) - set(binding)
    binding.update(izip(missing_names, repeat(MissingArgument)))
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
    def pos_args(self):
        warnings.warn("Use .args, not .pos_args")
        return self.args

    @property
    def arg(self):
        return _param_assignment(self)


def evaluate(p, instantiate_call=None, cache=None):
    """
    Evaluate a nested tree of functools.partial objects,
    used for deferred evaluation.

    Parameters
    ----------
    p : object
        If `p` is a partial, or a subclass of partial, it is
        expanded recursively. Otherwise, return.
    instantiate_call : callable, optional
        Rather than call `p.func` directly, instead call
        `instantiate_call(p.func, ...)`
    cache : dict, optional
        A result cache for resolving the exact same partial
        object more than once.

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
    instantiate_call = ((lambda f, *args, **kwargs: f(*args, **kwargs))
                        if instantiate_call is None else instantiate_call)
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
        obj, index = p.args
        if (isinstance(obj, functools.partial)
                and obj.func in (list, tuple)):
            index_val = evaluate(index, instantiate_call, cache)
            elem_val = evaluate(obj.args[index_val], instantiate_call, cache)
            try:
                int(index_val)
                cache[p] = elem_val
            except TypeError:
                # TODO: is this even conceivably used?
                cache[p] = obj.func(elem_val)
            return cache[p]

    # If we encounter switch(index, foo, bar, ...), don't
    # bother evaluating the branches we don't need.
    elif p.func == switch:
        assert p.keywords is None
        index, args = p.args[0], p.args[1:]
        index_val = evaluate(index, instantiate_call, cache)
        cache[p] = evaluate(p.args[index_val], instantiate_call, cache)
        return cache[p]

    args = [evaluate(arg, instantiate_call, cache) for arg in p.args]
    kw = [evaluate(kw, instantiate_call, cache)
          for kw in p.keywords] if p.keywords else {}
    # Cache the evaluated value (for subsequent calls that
    # will look at this cache dictionary) and return.
    cache[p] = instantiate_call(p, *args, **kw)
    return cache[p]
