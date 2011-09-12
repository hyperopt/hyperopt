"""

Random choice structure: independent factors.
- linear structure

sample construction maps random choices -> coherent sample
- tree or graph structure
- pointers to random choices

sampling means
rdist -> random choices -> sample

Whole process is modelled like this:

Test: maps a sample -> score and incurs a certain amount of cost.



"""

import copy, sys
import numpy
import bson
from bson import SON, BSON

import theano
from theano import tensor
import montetheano as MT


#
# Global methods
#

def SON2(*args, **kwargs):
    """
    Convenient SON-constructor
    """
    stringkeys = kwargs.pop('stringkeys', True)
    cls = kwargs.pop('cls', SON)
    if kwargs:
        raise TypeError('only stringkeys is valid kwarg', kwargs)
    if len(args)%2: raise ValueError('odd arg count to SON2', len(args))
    pairs = []
    for i in range(0,len(args),2):
        if stringkeys and not isinstance(args[i], (str,unicode)):
            raise TypeError('non-string key', args[i])
        pairs.append((args[i], args[i+1]))
    return cls(pairs)

#
# rdist constructors
#

distkey = '_dist2_'
rdist_registry = {}
class rdist_meta(type):
    def __new__(cls, name, bases, dct):
        if name in rdist_registry:
            raise Exception('name already registered as rdist', name)
        rval = type.__new__(cls, name, bases, dct)
        #print cls, name, bases, dct, rval
        rdist_registry[name] = rval
        return rval

def bless(son):
    """
    Modify the base types of a SON hierarchy in-place so that the SON nodes that
    used to be rdist instance become rdist instances again.
    """
    if distkey in son:
        base_type_name = son[distkey]
        base_type = rdist_registry[base_type_name]
        son.__class__ = base_type
        for c in son.children():
            bless(c)
    elif 'status' in son:
        # this is a trial base object
        bless(son['conf'])
    else:
        raise ValueError('what to do with this?', son)
    return son

def render(obj):
    if isinstance(obj, rdist):
        return obj.render()
    else:
        return obj

def rdistable(obj):
    return isinstance(obj, SON) and (distkey in obj)

class rdist(SON):
    __metaclass__ = rdist_meta
    def __new__(*args, **kwargs):
        #print 'new', args, kwargs
        return SON.__new__(*args, **kwargs)

    def flatten(self, rval=None):
        """Return all children recursively as list.
        """
        if rval is None:
            rval = []
        rval.append(self)
        for c in self.children():
            c.flatten(rval=rval)
        return rval

    def flatten_names(self, prefix='conf', rval=None):
        """
        Return the list of strings that indicate a path to a child node
        """
        if rval is None:
            rval = []
        rval.append(prefix)
        for c,cname in zip(self.children(), self.children_names()):
            c.flatten_names(prefix+cname, rval=rval)
        return rval

    def __init__(self):
        # no initial keys allowed because we're messing with the SON constructor
        super(rdist, self).__init__()
        self[distkey] = self.__class__.__name__
        self['choice'] = None

    def bless(self):
        """
        Replace any plain SON children with blessed ones.
        """
        pass

    def children(self):
        return []

    def children_names(self):
        return []

    def from_SON(self, son_template):
        raise NotImplementedError()

    def unsample(self):
        self['choice'] = None
        for child in self.children():
            child.unsample()

    def resample(self, rng):
        """Make random configuration choice for self.
        """
        for child in self.children():
            child.resample(rng)

    def render(self):
        """Return a sample from self (recursive).
        """
        raise NotImplementedError('override-me')

    def copy(self):
        raise Exception('are you sure you want to do a shallow copy?')

    def clone(self):
        return bless(copy.deepcopy(self))

    def sample(self, rng):
        """
        Make random choices for the nodes in template.
        """
        prior = bless(copy.deepcopy(self))
        #print type(self)
        #print type(prior)
        prior.unsample()
        prior.resample(rng)
        rval = prior.render()
        return rval

    # ---------------------------------------------------------------
    # --- TheanoBanditAlgo API
    # ---------------------------------------------------------------

    def theano_sampler(self, s_rng):
        """ Return Theano idxs, vals to sample from this rdist tree"""
        s_N = tensor.lscalar('s_N')
        if isinstance(s_rng, int):
            s_rng = MT.RandomStreams(s_rng)
        s_N = tensor.lscalar()
        s_elems = tensor.arange(s_N)
        memo = {}
        path = []
        self.theano_sampler_helper(s_rng, s_elems, memo, path)
        if len(memo) != len(self.flatten()):
            print memo
            print self
            assert (len(memo) == len(self.flatten()))
        idxs, vals = zip(*[memo[id(n_i)] for n_i in self.flatten()])
        return idxs, vals, s_N

    def theano_sampler_helper(self, s_rng, elems, memo, path):
        """
        Return a theano program that can draw a samples named in elems
        """
        assert id(self) not in memo # son graphs are tree-structured for now

        if 'geom' == self['_dist2_']:
            # TODO: use ranges!
            # TODO: cast to integer
            # TODO: match NIPS by using loguniform
            raise NotImplementedError('need a loguniform to match NIPS')
            vals = s_rng.lognormal(draw_shape=(elems.shape[0],), dtype='int32')
            memo[id(self)] = (elems, vals)

        elif 'expon' == self['_dist2_']:
            # TODO: use ranges!
            # TODO: match NIPS by using loguniform
            raise NotImplementedError('need a loguniform to match NIPS')
            vals = s_rng.lognormal(draw_shape=(elems.shape[0],))
            memo[id(self)] = (elems, vals)

        elif 'uniform' == self['_dist2_']:
            vals = s_rng.uniform(draw_shape=(elems.shape[0],),
                low=self['low'],
                high=self['high'])
            memo[id(self)] = (elems, vals)

        elif 'one_of' == self['_dist2_']:
            n_options = len(self['options'])
            #print 'n_options', n_options
            casevar = s_rng.categorical(
                        p=[1.0 / n_options] * n_options,
                        draw_shape=(elems.shape[0],))
            memo[id(self)] = (elems, casevar)
            for i, child in enumerate(self['options']):
                if child in self.children():
                    elems_i = elems[MT.for_theano.where(tensor.eq(i, casevar))]
                    child.theano_sampler_helper(s_rng, elems_i, memo,
                            path+[self])

        elif 'rSON' == self['_dist2_']:
            memo[id(self)] = (None, None)
            for child in self.children():
                child.theano_sampler_helper(s_rng, elems, memo, path + [self])
        else:
            raise ValueError(self)

    def idxs_vals_to_dict_list(self, idxs, vals):
        """ convert idxs, vals -> list-of-dicts"""
        nodes = self.flatten()
        idxdict = {}
        valdict = {}
        assert len(idxs) == len(vals)
        assert len(idxs) == len(nodes)
        for i, node in enumerate(nodes):
            idxdict[id(node)] = idxs[i]
            valdict[id(node)] = vals[i]

        # infer how many samples were drawn
        iii = []
        for idx_i in idxs:
            if idx_i is not None:
                try:
                    iii.extend(idx_i)
                except TypeError:
                    raise TypeError(idxs)

        rval = [self.nth_theano_sample(n, idxdict, valdict)
                for n in sorted(set(iii))]
        return rval

    def nth_theano_sample(self, n, idxdict, valdict):
        #XXX: move this into derived classes
        if 'geom' == self['_dist2_']:
            return valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]]
        elif 'expon' == self['_dist2_']:
            return valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]]
        elif 'uniform' == self['_dist2_']:
            return valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]]
        elif 'one_of' == self['_dist2_']:
            case = valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]]
            if self['options'][case] in self.children():
                return self['options'][case].nth_theano_sample(n, idxdict, valdict)
            else:
                return self['options'][case]
        elif 'rSON' == self['_dist2_']:
            rval = {}
            for (k, t) in self.items():
                if k not in self.special_keys:
                    if rdistable(t):
                        rval[k] = t.nth_theano_sample(n, idxdict, valdict)
                    else:
                        rval[k] = t
            return rval
        else:
            raise ValueError('rdist type not recognized', node['_dist2_'])


class rlist(rdist):
    def __init__(self, elements):
        super(rlist, self).__init__()
        self['elements'] = list(elements)
    if 0:
        @classmethod
        def new_from_SON(cls, son):
            self = cls(son['elements'])
            self.bless()
            self['choice'] = son['choice']
            return self

        def bless(self):
            for i, t in enumerate(self['elements']):
                if isinstance(t, SON) and distkey in t:
                    self['elements'][i] = bless(t)
    def children(self):
        return [t for t in self['elements'] if rdistable(t)]
    def children_names(self):
        return ['[%i]'%i for i in range(len(self.children()))]
    def render(self):
        return [render(elem) for elem in self['elements']]


def rlist2(*elements):
    return rlist(elements)


class rSON(rdist):
    special_keys = (distkey, 'choice')

    def __init__(self, *args, **kwargs):
        super(rSON, self).__init__()
        self.update(SON(*args, **kwargs))
        assert self[distkey] == 'rSON' # check here that it hasn't been changed

    def children(self):
        return [t for (k,t) in self.items() if rdistable(t)]

    def children_names(self):
        return ['.%s'%k for (k,t) in self.items() if rdistable(t)]

    def render(self):
        rval = SON()
        for key, val in self.items():
            if key in self.special_keys:
                continue
            try:
                rval[key] = render(val)
            except:
                print >> sys.stderr, "rSON render failed", key
                raise
        return rval


def rSON2(*args, **kwargs):
    assert 'cls' not in kwargs
    kwargs['cls'] = rSON
    return SON2(*((distkey, 'rSON') + args), **kwargs)


class one_of(rdist):
    def __init__(self, *options):
        super(one_of, self).__init__()
        self['options'] = list(options)

    if 0:
        @classmethod
        def new_from_SON(cls, son):
            self = cls(*son['options'])
            self.bless()
            self['choice'] = son['choice']
            return self

        def bless(self):
            for i, t in enumerate(self['options']):
                if isinstance(t, SON) and distkey in t:
                    self['options'][i] = bless(t)
    def children(self):
        return [t for t in self['options'] if rdistable(t)]
    def children_names(self):
        return ['{%i}'%i for i in range(len(self.children()))]

    def resample(self, rng):
        self['choice'] = choice = int(rng.randint(0, len(self['options'])))
        chosen = self['options'][choice]
        if isinstance(chosen, rdist):
            chosen.resample(rng)

    def render(self):
        #print self['options']
        #print self['choice']
        chosen = self['options'][self['choice']]
        return render(chosen)

    def n_options(self):
        return len(self['options'])


class LowHigh(rdist):
    def __init__(self, low, high):
        super(LowHigh, self).__init__()
        self['low'] = low
        self['high'] = high
    @classmethod
    def new_from_SON(cls, son):
        self = cls(son['low'], son['high'])
        self.bless()
        self['choice'] = son['choice']
        return self
    def render(self):
        return self['choice']


class uniform(LowHigh):
    def resample(self, rng):
        self['choice'] = float(rng.uniform(low=self['low'], high=self['high']))


class randint(LowHigh):
    def resample(self, rng):
        self['choice'] = int(rng.randint(low=self['low'], high=self['high']))


class expon(LowHigh):
    def __init__(self, low, high, zero=0):
        super(expon, self).__init__(low, high)
        self['zero'] = zero
        if low < zero or high < low:
            raise ValueError('range err', (zero, low, high))
    if 0:
        @classmethod
        def new_from_SON(cls, son):
            self = cls(son['low'], son['high'], son['zero'])
            self.bless()
            self['choice'] = son['choice']
            return self
    def resample(self, rng):
        low = self['low'] - self['zero']
        high = self['high'] - self['zero']
        self['choice'] = float(rng.uniform(low=numpy.log(low), high=numpy.log(high)))
    def render(self):
        return float(numpy.exp(self['choice']) + self['zero'])


class geom(expon):
    def __init__(self, low, high, zero=0, round=1):
        super(geom, self).__init__(low, high, zero=zero)
        self['round'] = round
    if 0:
        @classmethod
        def new_from_SON(cls, son):
            self = cls(son['low'], son['high'], son['zero'], son['round'])
            self.bless()
            self['choice'] = son['choice']
            return self
    def render(self):
        rval = super(geom, self).render()
        return (int(rval) // self['round']) * self['round']


#
#
# Testing
#
#


def test_basic():

    conf_template = rSON2(
        'LCN_width', one_of(5, 7, rlist2(2, 3, uniform(9,10))),
        'preprocessing', one_of(
            rlist2('GCN',),
            rlist2('LCN', rSON2(
                'width', one_of(5,7,9),
                )),
            rlist2('ZCA', rSON2(
                'n_examples', 10000,
                'filter_bias', expon(.01, 1.0),
                )),
            ),
        )

    print conf_template
    rng = numpy.random.RandomState(22)
    print conf_template.sample( rng)
    print conf_template.sample( rng)
    print conf_template.sample( rng)


def trial_print(trial, show_conf=True, show_results=True):
    print 'Trial (%s)'%trial.get('_id', 'no id')
    print '===================='
    print 'Status'
    print '------'
    print 'TODO: status info'
    print ''
    if show_conf:
        print 'Conf'
        print '----'
        for k,v in trial['conf'].items():
            print '  ', k, '=', v
        print ''
    if show_results:
        print 'Results'
        print '-------'
        for k,v in trial.get('results',{}).items():
            print '  ', k, '=', v


