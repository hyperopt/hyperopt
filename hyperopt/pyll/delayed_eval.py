"""
Delayed evaluation magic.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"


import functools
import inspect


def is_nested(frame=None):
    """
    Checks whether the calling scope of a frame is the definition scope.

    Parameters
    ----------
    frame : `frame` object
        Frame object, e.g. as returned by `inspect.currentframe()` or
        other related functions.

    Returns
    -------
    is_nested : bool
        `True` if the function being executed in `frame` was defined in
        the scope from which it was called, e.g. from the module scope
        if it was a top-level function, or from a function within which
        its definition is nested.
    """
    if frame is None:
        frame = inspect.currentframe().f_back
    caller = frame.f_back
    code = frame.f_code
    name = code.co_name
    return (caller is not None and name in caller.f_locals and
            getattr(caller.f_locals[name], 'func_code', None) == code)


def _resolve_upward(frame, name):
    """
    Check for a symbol in the `locals()` of calling scopes.

    Parameters
    ----------
    frame : `frame` object
        Frame object, e.g. as returned by `inspect.currentframe()` or
        other related functions.
    name : str
        The variable name to be resolved.

    Returns
    -------
    resolved : bool
        `True` if a binding was found (something like this is necessary,
        since a sentinel like `obj == None` could also be a valid bound
        value).
    obj : object
        The value to bind to `name`, if one was found, or `None`. Note that
        it can be `None` if `resolved` is `True`, as well, in which case
        `None` was the value found for `name` in some valid scope.
    """
    obj = None
    resolved = False
    while is_nested(frame):
        if name in frame.f_locals:
            resolved = True
            obj = frame.f_locals[name]
            break
        else:
            frame = frame.f_back
    return resolved, obj


class Delayed(object):
    """
    An object for which (nested) `getattr`s implement delayed evaluation.

    Parameters
    ----------
    proxy : callable
        `delayed.f(...)` will evaluate and return `proxy(f, ...)`.
        If unspecified, defaults to `functools.partial`. Provided object
        should mimic `functools.partial`'s interface.

    Notes
    -----
    TODO: examples

    TODO: make this picklable. Involves passing through a bunch of methods
    like `__getstate__`, `__setstate__`, but also `__mro__` and `__reduce__`
    and so forth.
    """
    def __init__(self, proxy=functools.partial):
        self._proxy_ = proxy

    def __getattribute__(self, name):
        caller = inspect.getouterframes(inspect.currentframe())[1][0]
        if name in ('__str__', '__repr__', '__dict__', '_proxy_'):
            return super(Delayed, self).__getattribute__(name)
        elif name in caller.f_locals:
            resolved = True
            obj = caller.f_locals[name]
        else:
            resolved = False
            if is_nested(caller):
                resolved, obj = _resolve_upward(caller, name)
            if not resolved and name in caller.f_globals:
                resolved = True
                obj = caller.f_globals[name]
            import __builtin__
            if not resolved and hasattr(__builtin__, name):
                resolved = True
                obj = getattr(__builtin__, name)
        if not resolved:
            raise NameError("name '%s' is not defined" % name)
        proxy = self._proxy_
        return DelayedObject(obj, proxy=proxy)


class DelayedObject(object):
    """
    An object that wraps attribute lookups on an object.

    Parameters
    ----------
    proxy : callable
        `delayed.f(...)` will evaluate and return `proxy(f, ...)`.
        If unspecified, defaults to `functools.partial`. Provided object
        should mimic `functools.partial`'s interface.

    Notes
    -----
    This makes `__getattribute__` return an instance of `DelayedObject` for any

    TODO: examples

    TODO: make this picklable. Involves passing through a bunch of methods
    like `__getstate__`, `__setstate__`, but also `__mro__` and `__reduce__`
    and so forth.
    """
    def __init__(self, obj, proxy=functools.partial):
        self._obj_ = obj
        self._proxy_ = proxy

    def __call__(self, *args, **kwargs):
        return self._proxy_(self._obj_, *args, **kwargs)

    def __getattribute__(self, name):
        if name == '__call__':
            return self.__call__
        elif name in ('__str__', '__repr__', '__dict__', '_obj_', '_proxy_'):
            return super(DelayedObject, self).__getattribute__(name)
        # TODO: figure out how this plays when self._obj_ is a `DelayedObject`
        # or something evil. Also, is the else clause actually the right thing?
        elif name in self._obj_.__dict__:
            if hasattr(self._obj_, name):
                return DelayedObject(getattr(self._obj_, name), self._proxy_)
            else:
                raise AttributeError(name)
        else:
            return super(DelayedObject, self).__getattribute__(name)
