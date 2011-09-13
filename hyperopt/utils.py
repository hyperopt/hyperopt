import sys
import logging
logger = logging.getLogger(__name__)

def import_tokens(tokens):
    # XXX Document me
    # import as many as we can
    rval = None
    for i in range(len(tokens)):
        modname = '.'.join(tokens[:i+1])
        # XXX: try using getattr, and then merge with load_tokens
        try:
            logger.info('importing %s' % modname)
            exec "import %s" % modname
            exec "rval = %s" % modname
        except ImportError, e:
            logger.info('failed to import %s' % modname)
            logger.info('reason: %s' % str(e))
            break
    return rval, tokens[i:]

def load_tokens(tokens):
    # XXX: merge with import_tokens
    logger.info('load_tokens: %s' % str(tokens))
    symbol, remainder = import_tokens(tokens)
    for attr in remainder:
        symbol = getattr(symbol, attr)
    return symbol


def json_lookup(json):
    symbol = load_tokens(json.split('.'))
    return symbol


def json_call(json):
    """
    Return a dataset class instance based on a string, tuple or dictionary

    .. code-block:: python

        iris = json_call('datasets.toy.Iris')

    This function works by parsing the string, and calling import and getattr a
    lot. (XXX)

    """
    if isinstance(json, basestring):
        symbol = json_lookup(json)
        return symbol()
    elif isinstance(json, dict):
        raise NotImplementedError('dict calling convention undefined', json)
    elif isinstance(json, (tuple, list)):
        raise NotImplementedError('seq calling convention undefined', json)
    else:
        raise TypeError(json)



