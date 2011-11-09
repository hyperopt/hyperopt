"""
Utilities for converting between genson descriptions to theano samplers.
Basic idea is to create a gDist object from a genson string; and then pass
the gDist object as a template to a (theano) bandit.
"""
import copy
import logging
import sys

import numpy
import bson
from bson import SON, BSON
import genson
import genson.parser
from genson.util import isdict, set_global_seed
import theano
from theano import tensor
import montetheano as MT

import base


class SetOp(theano.Op):
    """
    Inputs: N vectors of integers.
    Returns: sorted union of all input elements.
    """
    def __init__(self, operation):
        self.operation = operation

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.operation == other.operation)

    def __hash__(self):
        return hash((type(self), self.operation))

    def make_node(self, *argv):
        argv = map(tensor.as_tensor_variable, argv)
        if len(argv) == 0:
            raise TypeError('union requires at least one argument')
        for v in argv:
            if v.ndim != 1:
                raise TypeError('1d symbolic array required', v)
            if 'int' not in str(v.dtype):
                raise TypeError('int dtype required', v)
        return theano.gof.Apply(self,
                list(argv),
                [argv[0].type()])

    def perform(self, node, inputs, outstorage):
        for ii in inputs:
            if 'int' not in str(ii.dtype):
                raise TypeError('non-int arg to SetOp', ii)
        ans = set(inputs[0])
        if self.operation == 'union':
            ans.update(*inputs[1:])
        elif self.operation == 'difference':
            ans.difference_update(*inputs[1:])
        else:
            raise NotImplementedError(self.operation)
        npy_ans = numpy.array(sorted(ans), dtype=node.outputs[0].dtype)
        if npy_ans.ndim != 1:
            raise TypeError('didnt make array', ans)
        outstorage[0][0] = node.outputs[0].type.filter(npy_ans, strict=True)


union = SetOp('union')
#XXX: s/union/set_union
set_difference = SetOp('difference')


def gdistable(obj):
    return isinstance(obj, SON)


def get_obj_type(t):
    if isinstance(t, genson.internal_ops.GenSONBinaryOp):
        return gBinOp
    elif isinstance(t, genson.functions.GenSONFunction):
        return gFunc
    elif isinstance(t, genson.functions.GaussianRandomGenerator):
        return gGauss
    elif isinstance(t, genson.functions.UniformRandomGenerator):
        return gUniform
    elif isinstance(t, genson.functions.ChoiceRandomGenerator):
        return gChoice
    elif isinstance(t, genson.functions.RandintGenerator):
        return gRandint
    elif isinstance(t, genson.functions.LognormalRandomGenerator):
        return gLognormal
    elif isinstance(t, genson.functions.QuantizedLognormalRandomGenerator):
        return gQLognormal
    elif isinstance(t, genson.references.ScopedReference):
        return gRef
    elif hasattr(t, 'keys'):
        return gDict
    elif hasattr(t, '__iter__'):
        return gList
    else:
        return pass_through


def pass_through(t, path=None):
    return t


def get_obj(t, path):
    return get_obj_type(t)(t, path=path)


class gSON(SON):
    """Base class for translation between genson objects and theano samplers.
    The class is never directly called, all instances of it are instances of
    subclasses defined below.   See class gDist below for main "entry point"
    for creating gDist objects.
    """

    def __new__(*args, **kwargs):
        #print 'new', args, kwargs
        return SON.__new__(*args, **kwargs)

    def __init__(self, genson_obj, path=[]):
        super(gSON, self).__init__()
        self.genson_obj = genson_obj
        self.path = path
        self.make_contents()

    def make_contents(self):
        if hasattr(self, 'params'):
            for k in self.params:
                setattr(self, k, get_obj(getattr(self.genson_obj, k),
                                         self.path))

    def children(self):
        return [getattr(self, x) for x in self.params \
                          if gdistable(getattr(self, x))]

    def children_names(self):
        return [x for x in self.params if gdistable(getattr(self, x))]

    def render(self):
        pass

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
        """Return the list of strings that indicate a path to a child node
        """
        if rval is None:
            rval = []
        rval.append(prefix)
        for c, cname in zip(self.children(), self.children_names()):
            c.flatten_names(prefix + cname, rval=rval)
        return rval

    def random_nodes(self):
        return [node for node in self.flatten() if isinstance(node, gRandom)]

    def get_elems(self, s_rng, elems, memo):
        for child in self.children():
            child.get_elems(s_rng, elems, memo)
        memo[id(self)] = elems

    def correct_elems(self, memo, corrected_memo):
        if not id(self) in corrected_memo:
            to_correct = self.children()
            if hasattr(self, 'referenced_by'):
                to_correct += self.referenced_by
            for child in to_correct:
                child.correct_elems(memo, corrected_memo)
            elems = memo.pop(id(self))
            if elems is not None and hasattr(self, 'referenced_by'):
                for c in self.referenced_by:
                    ec = corrected_memo[id(c)]
                    elems = union(elems, ec)
            corrected_memo[id(self)] = elems

    def nth_sample(self, k, n, idxdict, valdict):
        if k in self.children():
            return k.nth_theano_sample(n, idxdict, valdict)
        else:
            return k

    def nth_theano_sample(self, n, idxdict, valdict):
        v = numpy.where(idxdict[id(self)] == n)[0][0]
        return valdict[id(self)][v].tolist()


class gList(gSON):
    """List of elements that can be either constant or gdist
    """

    def make_contents(self):
        self['elements'] = [get_obj(t, self.path) for t in self.genson_obj]

    def children(self):
        return [t for t in self['elements'] if gdistable(t)]

    def children_names(self):
        return ['[%i]' % i for i in range(len(self.children()))]

    def render(self):
        return [render(elem) for elem in self['elements']]

    def theano_sampler_helper(self, memo, s_rng):
        for t in self.children():
            t.theano_sampler_helper(memo, s_rng)
        vals = [get_value(t, memo) for t in self['elements']]
        memo[id(self)] = (memo[id(self)], vals)

    def nth_theano_sample(self, n, idxd, vald):
        return [self.nth_sample(t, n, idxd, vald) for t in self['elements']]


class gDict(gSON):
    """Dictionary mapping strings to either constant or gdist
    """

    def make_contents(self):
        for k, t in self.genson_obj.items():
            self[k] = get_obj(t, self.path + [self])

    def children(self):
        return [t for (k, t) in self.items() if gdistable(t)]

    def children_names(self):
        return ['.%s' % k for (k, t) in self.items() if gdistable(t)]

    def theano_sampler_helper(self, memo, s_rng):
        for t in self.children():
            t.theano_sampler_helper(memo, s_rng)
        val_dict = dict([(k, get_value(t, memo)) for k, t in self.items()])
        memo[id(self)] = (memo[id(self)], val_dict)

    def nth_theano_sample(self, n, idxd, vald):
        return dict([(k, self.nth_sample(t, n, idxd, vald)) \
                                          for (k, t) in self.items()])


class gDist(gDict):
    """Main entry point for creating theano bandit templates from genSON
    objects.   Basically gDist objects are just gDict objects which have (1)
    a facility for calling the GENSON object parser in the __init__ and
    (2) support sampling methods, both for regular (non-theano) bandits as well
    as theano-based sampling.
    """

    def __init__(self, genson_string):
        parser = genson.parser.GENSONParser()
        genson_obj = parser.parse_string(genson_string)
        super(gDist, self).__init__(genson_obj)
        self.genson_generator = genson.JSONGenerator(genson_obj)

    def seed(self, seed):
        """ Reset the generators used by `self.sample` """
        try:
            self.genson_generator.seed(seed)
        except AttributeError:
            # -- currently there is no such function
            #    self.genson_generator.seed()
            #    But this is what it should probably do
            seedgen = numpy.random.RandomState(seed)
            for g in self.genson_generator.generators:
                g.seed(int(seedgen.randint(2 ** 31)))

    def sample(self, seed=None):
        """ Return next random configuration template """
        if seed is not None:
            self.seed(seed)
        for g in self.genson_generator.generators:
            g.advance()
        return genson.resolve(self.genson_generator.genson_dict)

    def theano_sampler(self, s_rng):
        """ Return Theano idxs, vals to sample from this rdist tree"""
        s_N = tensor.lscalar('s_N')
        if isinstance(s_rng, int):
            s_rng = MT.RandomStreams(s_rng)
        s_N = tensor.lscalar()
        elems = tensor.arange(s_N)
        memo = {}
        self.get_elems(s_rng, elems, memo)
        corrected_memo = {}
        self.correct_elems(memo, corrected_memo)
        memo = corrected_memo
        self.theano_sampler_helper(memo, s_rng)
        rnodes = self.random_nodes()
        idxs, vals = zip(*[memo[id(n)] for n in rnodes])
        return idxs, vals, s_N

    def idxs_vals_to_dict_list(self, idxs, vals):
        """ convert idxs, vals -> list-of-dicts"""
        nodes = self.random_nodes()
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
        BSON(rval)  # make sure encoding is possible
        return rval


class gBinOp(gSON):
    params = ['a', 'b']

    op_dict = dict([('+', tensor.add),
                    ('-', tensor.sub),
                    ('*', tensor.mul),
                    ('/', tensor.div_proxy),
                    ('**', tensor.pow)])

    def make_contents(self):
        super(gBinOp, self).make_contents()
        self.op = self.op_dict[self.genson_obj.op]

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        a = get_value(self.a, memo)
        b = get_value(self.b, memo)
        memo[id(self)] = [memo[id(self)], self.op(a, b)]

    def nth_theano_sample(self, n, idxdict, valdict):
        asample = self.nth_sample(self.a, n, idxdict, valdict)
        bsample = self.nth_sample(self.b, n, idxdict, valdict)
        return self.op.nfunc(asample, bsample)


class gFunc(gSON):
    params = ['args', 'kwargs']

    def make_contents(self):
        super(gFunc, self).make_contents()
        self.func = getattr(tensor, self.genson_obj.name)

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        args = tuple(get_value(self.args, memo))
        kwargs = get_value(self.kwargs, memo)
        memo[id(self)] = [memo[id(self)], self.func(*args, **kwargs)]

    def nth_theano_sample(self, n, idxdict, valdict):
        argsample = tuple(self.args.nth_theano_sample(n, idxdict, valdict))
        kwargsample = self.kwargs.nth_theano_sample(n, idxdict, valdict)
        return self.func.nfunc(*argsample, **kwargsample)


class gRef(gSON):

    params = []

    def make_contents(self):
        scope_list = self.genson_obj.scope_list
        self.reference = resolve_scoped_reference(scope_list[:],
                                                  self.path[:])
        self.propagate_up(self.reference)

    def propagate_up(self, reference):
        if isinstance(reference, gSON):
            if hasattr(reference, 'referenced_by'):
                reference.referenced_by.append(self)
            else:
                reference.referenced_by = [self]

    def get_elems(self, s_rng, elems, memo):
        if isinstance(self.reference, gSON):
            memo[id(self)] = memo[id(self.reference)]
        else:
            memo[id(self)] = None

    def theano_sampler_helper(self, memo, s_rng):
        if isinstance(self.reference, gSON):
            memo[id(self)] = memo[id(self.reference)]
        else:
            memo[id(self)] = (memo[id(self)], self.reference)

    def nth_theano_sample(self, n, idxdict, valdict):
        return self.reference.nth_theano_sample(n, idxdict, valdict)


def resolve_scoped_reference(ref, path):
    """
    """

    if len(path) == 0:
        # TODO: better
        raise Exception("Invalid reference")

    # pop an element off of the scope list
    element_to_resolve = ref.pop(0)

    if(element_to_resolve == 'root'):
        return resolve_scoped_reference(ref, [path[0]])

    if(element_to_resolve == 'parent'):
        return resolve_scoped_reference(ref, path[0:-1])

    if(element_to_resolve == 'this'):
        return resolve_scoped_reference(ref, path)

    current_context = path[-1]

    if not isdict(current_context):
        raise Exception("Invalid reference")

    if not element_to_resolve in current_context:
        # TODO: make better
        raise Exception("Unknown key: %s" % element_to_resolve)

    resolved_element = current_context[element_to_resolve]

    if len(ref) != 0:
        path.append(resolved_element)
        return resolve_scoped_reference(ref, path)

    return resolved_element


class gRandom(gSON):
    def get_shape(self, size, elems):
        if isinstance(size, int):
            if size == 1:
                ds = (elems.shape[0],)
            else:
                ds = (elems.shape[0], size)
        else:
            ds = (elems.shape[0],) + tuple(size)

        return ds


class gGauss(gRandom):
    params = ['mean', 'stdev', 'size']

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        mu = get_value(self.mean, memo)
        stdev = get_value(self.stdev, memo)
        self.sigma = sigma = stdev ** 2
        elems = memo[id(self)]
        size = get_value(self.size, memo)
        ds = get_size(size,elems)
        vals = s_rng.normal(draw_shape=ds,
                            mu=mu, sigma=sigma)
        memo[id(self)] = (elems, vals)


class gLognormal(gRandom):
    params = ['mean', 'stdev', 'size']

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        mu = get_value(self.mean, memo)
        stdev = get_value(self.stdev, memo)
        sigma = self.sigma = stdev ** 2
        elems = memo[id(self)]
        size = get_value(self.size, memo)
        ds = get_size(size,elems)
        vals = s_rng.lognormal(draw_shape=ds,
                            mu=mu, sigma=sigma)
        memo[id(self)] = (elems, vals)


class gQLognormal(gRandom):
    params = ['mean', 'stdev', 'size', 'round']

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        mu = get_value(self.mean, memo)
        stdev = get_value(self.stdev, memo)
        self.sigma = sigma = stdev ** 2
        elems = memo[id(self)]
        size = get_value(self.size, memo)
        round = get_value(self.round, memo)
        ds = get_size(size,elems)
        vals = s_rng.quantized_lognormal(draw_shape=ds, mu=mu, sigma=sigma, 
                                         step = round, dtype = 'int64')
        memo[id(self)] = (elems, vals)


class gUniform(gRandom):
    params = ['min', 'max', 'size']

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        low = get_value(self.min, memo)
        high = get_value(self.max, memo)
        size = get_value(self.size, memo)
        elems = memo[id(self)]
        ds = get_size(size,elems)
        vals = s_rng.uniform(draw_shape=ds,
                             low=low, high=high)
        memo[id(self)] = (elems, vals)


class gChoice(gRandom):

    def make_contents(self):
        self.vals = [get_obj(t, self.path) for t in self.genson_obj.vals]

    def children(self):
        return [x for x in (self.vals) if gdistable(x)]

    def children_names(self):
        return ['[%i]' % i for i in range(len(self.children()))]

    def get_elems(self, s_rng, elems, memo):
        n_options = len(self.vals)
        casevar = s_rng.categorical(
                    p=[1.0 / n_options] * n_options,
                    draw_shape=(elems.shape[0],))
        self.casevar = casevar
        memo[id(self)] = elems
        for i, child in enumerate(self.vals):
            if child in self.children():
                elems_i = elems[MT.for_theano.where(tensor.eq(i, casevar))]
                child.get_elems(s_rng, elems_i, memo)

    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        n_options = len(self.vals)
        elems = memo[id(self)]
        memo[id(self)] = (elems, self.casevar)

    def nth_theano_sample(self, n, idxdict, valdict):
        v = numpy.where(idxdict[id(self)] == n)[0][0]
        case = valdict[id(self)][v].tolist()
        if self.vals[case] in self.children():
            return self.vals[case].nth_theano_sample(n, idxdict, valdict)
        else:
            return self.vals[case]


class gRandint(gRandom):

    params = ['min','max','size']
    
    def theano_sampler_helper(self, memo, s_rng):
        low = get_value(self.min, memo)
        high = get_value(self.max, memo)
        n_options = int(high-low)
        size = get_value(self.size, memo)
        elems = memo[id(self)]
        ds = get_size(size,elems)
        casevar = s_rng.categorical(
                    p=[1.0 / n_options] * n_options,
                    draw_shape=ds)       
        elems = memo[id(self)]
        memo[id(self)] = (elems, casevar)
  
    def nth_theano_sample(self, n, idxdict, valdict):
        v = numpy.where(idxdict[id(self)] == n)[0][0]
        case = valdict[id(self)][v].tolist()
        return self.unroll(case, n, idxdict, valdict)

    def unroll(self, case, n, idxdict, valdict):
        if not hasattr(case, '__iter__'):
            if self.vals[case] in self.children():
                return self.vals[case].nth_theano_sample(n, idxdict, valdict)
            else:
                return self.vals[case]
        else:
            return [self.unroll(c, n, idxdict, valdict) for c in case]
            

def get_size(size,elems):
    if isinstance(size, int):
        if size == 1:
            ds = (elems.shape[0],)
        else:
            ds = (elems.shape[0], size)
    else:
        ds = (elems.shape[0],) + tuple(size)
    return ds

def get_value(x, memo):
    if isinstance(x, gSON):
        return memo[id(x)][1]
    else:
        return x


class GensonBandit(base.Bandit):
    def __init__(self, genson_file):
        template = gDist(open(genson_file).read())
        base.Bandit.__init__(self.template)
