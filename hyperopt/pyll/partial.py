"""
Support code for `functools.partial` based deferred-evaluation
mechanism.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

from collections import deque
import compiler
from functools import partial as _partial
import warnings
from itertools import izip, repeat

import networkx as nx

# TODO: support o_len functionality from old Apply nodes


def _list_node(*args):
    """
    Wrapper for the builtin `list()` that calls it on *args.

    Handles tuples encountered by as_partialplus, so that we don't
    have to have special logic for recursing on
    """
    return list(args)


def _tuple_node(*args):
    """
    Wrapper for the builtin `tuple()` that calls it on *args.
    """
    return tuple(args)


def _call_with_list_of_pos_args(f, *args):
    return f(args)


def partial(f, *args, **kwargs):
    """
    A workalike for `functools.partial` that actually (recursively)
    creates `PartialPlus` objects via `as_partialplus`.

    Parameters
    ----------
    f : callable
        Function whose evaluation to defer.

    Notes
    -----
    Remaining positional and keyword arguments are passed along to
    `f`, as `functools.partial`.
    """
    return as_partialplus(_partial(f, *args, **kwargs))


def as_partialplus(p):
    """
    Convert a (possibly nested) `partial` to the

    Parameters
    ----------
    p : object
        If `p` is a `functools.partial`, a list, or a tuple, it is
        given special treatment, and its arguments/elements are recursed
        upon. Otherwise, it is wrapped in a `Literal`.

    Returns
    -------
    node : object
        A `PartialPlus`, or a `Literal`.
    """
    if isinstance(p, (PartialPlus, Literal)):
        return p
    elif isinstance(p, _partial):
        args = [as_partialplus(a) for a in p.args]
        if p.keywords:
            kwargs = dict((k, as_partialplus(v))
                          for k, v in p.keywords.iteritems())
            return PartialPlus(p.func, *args, **kwargs)
        else:
            return PartialPlus(p.func, *args)
    # Not using isinstance, on purpose. Want literal lists and tuples,
    # not subclasses.
    elif type(p) in (list, tuple):
        if type(p) == list:
            func = _list_node
        else:
            func = _tuple_node
        return PartialPlus(func, *(as_partialplus(e) for e in p))
    # Definitely want this to work for OrderedDicts.
    elif isinstance(p, dict):
        # Special-case dictionaries to recurse on values.
        # TODO: recurse on keys?
        args = [Literal(p.__class__)] + [as_partialplus((k, v))
                                         for k, v in p.iteritems()]
        return PartialPlus(_call_with_list_of_pos_args, *args)
    else:
        return Literal(p)


def depth_first_traversal(root):
    """
    Perform a depth-first traversal of a graph of PartialPlus objects.

    Parameters
    ----------
    root : object
        A `Node` object.

    Returns
    -------
    sequence : list
        A list of `functools.partial` nodes from the graph,
        in a depth-first order.
    """
    assert isinstance(root, Node)
    visited = {}
    to_visit = deque()
    to_visit.append(root)
    while len(to_visit) > 0:
        node = to_visit.pop()
        if node not in visited:
            visited[node] = True
            yield node
            if isinstance(node, PartialPlus):
                children = node.args + (tuple(node.keywords.values())
                                        if node.keywords is not None else ())
                for c in children:
                    to_visit.append(c)


def topological_sort(expr):
    """
    Return the nodes of `expr` sub-tree in topological order.

    Raises networkx.NetworkXUnfeasible if subtree contains cycle.
    """
    # TODO: remove networkx dependency. It should be short enough.
    G = nx.DiGraph()
    for node in depth_first_traversal(expr):
        G.add_edges_from([(n_in, node) for n_in in node.inputs()])
    order = nx.topological_sort(G)
    assert order[-1] == expr
    return order


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
        binding.setdefault(param_i, Literal(default_i))

    # -- mark any outstanding parameters as missing
    missing_names = set(params) - set(binding)
    binding.update(izip(missing_names, repeat(MissingArgument)))
    return binding


class Node(object):
    def clone(self):
        bindings = {}
        nodes = topological_sort(self)
        for node in nodes:
            if isinstance(node, Literal):
                bindings[node] = Literal(node.value)
            else:  # PartialPlus
                func = node.func
                args = [bindings[a] for a in node.args]
                keywords = dict((k, bindings[v])
                                for k, v in node.keywords.iteritems())
                bindings[node] = PartialPlus(func, *args, **keywords)
        return bindings[nodes[-1]]

    def inputs(self):
        return ()


class Literal(Node):
    func = None
    args = None
    keywords = None

    def __init__(self, value):
        self._value = value

    def __gt__(self, other):
        return self.value > other.value

    def __lt__(self, other):
        if not hasattr(other, 'value'):
            return False
        return self.value < other.value

    def __eq__(self, other):
        if not hasattr(other, 'value'):
            return False
        return self.value == other.value

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return None


class PartialPlus(_partial, Node):
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

    def __init__(self, f, *args, **kwargs):
        assert all(isinstance(a, Node) for a in args)
        assert all(isinstance(v, Node) for k, v in kwargs.iteritems())
        super(PartialPlus, self).__init__(self, f, *args, **kwargs)
        self._keywords = kwargs
        self._args = args

    def __call__(self, *args, **kwargs):
        raise TypeError("use evaluate() for %s objects" %
                        partial.__name__)

    def __add__(self, other):
        return partial(_binary_arithmetic, self, other, '+')

    def __sub__(self, other):
        return partial(_binary_arithmetic, self, other, '-')

    def __mul__(self, other):
        return partial(_binary_arithmetic, self, other, '*')

    def __floordiv__(self, other):
        return partial(_binary_arithmetic, self, other, '//')

    def __mod__(self, other):
        return partial(_binary_arithmetic, self, other, '%')

    def __divmod__(self, other):
        return partial(divmod, self, other)

    def __pow__(self, other, modulo=None):
        return partial(pow, self, other, modulo)

    def __lshift__(self, other):
        return partial(_binary_arithmetic, self, other, '<<')

    def __rshift__(self, other):
        return partial(_binary_arithmetic, self, other, '>>')

    def __and__(self, other):
        return partial(_binary_arithmetic, self, other, '&')

    def __xor__(self, other):
        return partial(_binary_arithmetic, self, other, '^')

    def __or__(self, other):
        return partial(_binary_arithmetic, self, other, '|')

    def __div__(self, other):
        return partial(_binary_arithmetic, self, other, '/')

    def __truediv__(self, other):
        return partial(_binary_arithmetic, self, other, '/')

    def __lt__(self, other):
        return partial(_binary_arithmetic, self, other, '<')

    def __le__(self, other):
        return partial(_binary_arithmetic, self, other, '<=')

    def __eq__(self, other):
        return partial(_binary_arithmetic, self, other, '==')

    def __ne__(self, other):
        return partial(_binary_arithmetic, self, other, '!=')

    def __gt__(self, other):
        return partial(_binary_arithmetic, self, other, '>')

    def __ge__(self, other):
        return partial(_binary_arithmetic, self, other, '>=')

    def __neg__(self):
        return partial(_unary_arithmetic, self, '-')

    def __pos__(self):
        return partial(_unary_arithmetic, self, '+')

    def __abs__(self):
        return partial(abs, self)

    def __invert__(self):
        return partial(abs, self)

    def __complex__(self):
        return partial(complex, self)

    def __int__(self):
        return partial(int, self)

    def __long__(self):
        return partial(long, self)

    def __float__(self):
        return partial(float, self)

    def __oct__(self):
        return partial(oct, self)

    def __hex__(self):
        return partial(hex, self)

    def __getitem__(self, item):
        if not isinstance(item, Node):
            item = as_partialplus(item)
        return partial(_getitem, self, item)

    @property
    def pos_args(self):
        warnings.warn("Use .args, not .pos_args")
        return self.args

    @property
    def name(self):
        # TODO: should we rewrite the name matching stuff in terms of function
        # identity?
        if hasattr(self.func, '__name__'):
            return self.func.__name__
        else:
            return self.func.func_name

    def inputs(self):
        # TODO: make this a property
        return self.args + (tuple(self.keywords.itervalues())
                            if self.keywords is not None else ())

    @property
    def arg(self):
        # TODO: bindings
        return _param_assignment(self)

    def replace_input(self, old_node, new_node):
        new_args = tuple(new_node if obj is old_node else obj
                         for obj in self.args)
        new_keywords = [(key, new_node) if val is old_node else (key, val)
                        for key, val in self.keywords.iteritems()]
        return partial(self.func, *new_args, **new_keywords)

    @property
    def keywords(self):
        """
        Overwrite the default keywords attribute to always have a dictionary
        in that spot rather than None sometimes, which makes for a lot of
        annoying special cases.
        """
        return self._keywords

    @property
    def args(self):
        """
        Overwrite the default args attribute so that we have more control
        over it, and can thereby append arguments.
        """
        return self._args

    def append_arg(self, arg):
        self._args = self._args + (arg,)

    def extend_args(self, args):
        self._args = self._args + tuple(args)


def evaluate(p, instantiate_call=None, bindings=None):
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
    bindings : dict, optional
        A dictionary mapping `Node` objects to values to use
        in their stead. Used to cache objects already evaluated.

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
    bindings = {} if bindings is None else bindings

    # If we've encountered this exact partial node before,
    # short-circuit the evaluation of this branch and return
    # the pre-computed value.
    if p in bindings:
        return bindings[p]
    if isinstance(p, Literal):
        bindings[p] = p.value
        return bindings[p]

    # When evaluating an expression of the form
    # `list(...)[item]`
    # only evaluate the element(s) of the list that we need.
    if p.func == _getitem:
        obj, index = p.args
        if (isinstance(obj, _partial)
                and obj.func in (_list_node, _tuple_node)):
            index_val = evaluate(index, instantiate_call, bindings)
            # TODO: is_iterable
            elem_val = obj.args[index_val]
            if isinstance(index_val, slice):  # TODO: something more robust?
                elem_val = obj.func(*[evaluate(e, instantiate_call, bindings)
                                      for e in elem_val])
            else:
                elem_val = evaluate(elem_val, instantiate_call, bindings)
            try:
                # bindings the value of this subexpression as
                int(index_val)
                bindings[p] = elem_val
            except TypeError:
                # TODO: is this even conceivably used?
                bindings[p] = instantiate_call(p.func, elem_val, index_val)
            return bindings[p]

    args = [evaluate(arg, instantiate_call, bindings) for arg in p.args]
    kw = dict((kw, evaluate(val, instantiate_call, bindings))
              for kw, val in p.keywords.iteritems()) if p.keywords else {}
    # bindings the evaluated value (for subsequent calls that
    # will look at this bindings dictionary) and return.
    bindings[p] = instantiate_call(p.func, *args, **kw)
    return bindings[p]
