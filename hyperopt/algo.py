
def register_suggest(fn, name=None):
    if name is None:
        name = fn.__name__
    globals()[name] = fn
