"""
Sample problems on which to test algorithms.

"""
import copy
import logging
import sys

import numpy
import bson
from bson import SON, BSON
import genson
import genson.parser
from genson.util import isdict
import theano
from theano import tensor
import montetheano as MT

import base


class Union(theano.Op):
    """

    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, v1, v2):
        v1 = tensor.as_tensor_variable(v1)
        v2 = tensor.as_tensor_variable(v2)
        if v1.ndim != 1: 
            raise TypeError()
        if v2.ndim != 1: 
            raise TypeError()            
        assert 'int' in str(v1.dtype)
        return theano.gof.Apply(self,
                [v1, v2],
                [v1.type()])

    def perform(self, node, inputs, outstorage):
        v1, v2 = inputs # numeric!
        ans = np.array(sorted(list(set(v1).union(v2)))).astype(v1.dtype)
        outstorage[0][0] = ans

    #XXX: infer_shape
union = Union()
        
def gdistable(obj):
    return isinstance(obj, SON)
    
def get_obj_type(t):
    if isinstance(t,genson.internal_ops.GenSONBinaryOp):
        return gBinOp
    elif isinstance(t,genson.functions.GenSONFunction):
        return gFunc
    elif isinstance(t,genson.functions.GaussianRandomGenerator):
        return gGauss
    elif isinstance(t,genson.functions.UniformRandomGenerator):
        return gUniform
    elif isinstance(t,genson.functions.ChoiceRandomGenerator):
        return gChoice
    elif isinstance(t,genson.references.ScopedReference):
        return gRef
    elif hasattr(t,'keys'):
        return gDict
    elif hasattr(t,'__iter__'):
        return gList
    else:
        return pass_through
        
def pass_through(t,path=None):
   return t
   
def get_obj(t,path):
    return get_obj_type(t)(t,path=path)


class gSON(SON):

    def __new__(*args, **kwargs):
        #print 'new', args, kwargs
        return SON.__new__(*args, **kwargs)
        
    def __init__(self,genson_obj,path=[]):
        super(gSON,self).__init__()
        self.genson_obj = genson_obj
        self.path = path
        self.make_contents()

    def make_contents(self):
        if hasattr(self,'params'):
            for k in self.params:
                setattr(self,k,get_obj(getattr(self.genson_obj,k),self.path+[self]))
    
    def children(self):
        return [getattr(self,x) for x in self.params if gdistable(getattr(self,x))]
        
    def children_names(self):
        return [x for x in self.params if gdistable(getattr(self,x))]
    
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
        """
        Return the list of strings that indicate a path to a child node
        """
        if rval is None:
            rval = []
        rval.append(prefix)
        for c,cname in zip(self.children, self.children_names):
            c.flatten_names(prefix+cname, rval=rval)
        return rval
        
    def get_elems(self,s_rng,elems,memo):
        for child in self.children():
            child.get_elems(s_rng,elems,memo)
        memo[id(self)] = elems
        
    def correct_elems(self,memo,corrected_memo):
        if not id(self) in corrected_memo:
            to_correct = self.children()
            if hasattr(self,'referenced_by'):
                to_correct += self.referenced_by                  
            for child in to_correct:
                child.correct_elems(memo,corrected_memo)
            elems = memo.pop(id(self))
            if elems is not None and hasattr(self,'referenced_by'):
                for c in self.referenced_by:
                    ec,ev = corrected_memo[id(c)]
                    elems = union(elems,ec) 
            corrected_memo[id(self)] = elems
            
                    
    def nth_sample(self,k,n,idxdict, valdict):
        if k in self.children():
            return k.nth_theano_sample(n, idxdict, valdict)
        else:
            return k


class gList(gSON):
    """List of elements that can be either constant or gdist
    """
    
    def make_contents(self):
        self['elements'] = [get_obj(t,self.path+[self]) for t in self.genson_obj]
        
    def children(self):
        return [t for t in self['elements'] if gdistable(t)]

    def children_names(self):
        return ['[%i]'%i for i in range(len(self.children()))]

    def render(self):
        return [render(elem) for elem in self['elements']]

    def theano_sampler_helper(self, memo, s_rng):
        for t in self.children():
            t.theano_smapler_helper(memo, s_rng)
        vals = [get_value(t,memo) for t in self['elements']]
        memo[id(self)] = (memo[id(self)],vals)

    def nth_theano_sample(self, n, idxdict, valdict):
        return [self.nth_sample(t,n,idxdict,valict) for t in self['elements']]
    
        
class gDict(gSON):
    """Dictionary mapping strings to either constant or gdist
    """

    def make_contents(self):
        for k,t in self.genson_obj.items():
            self[k] = get_obj(t,self.path+[self])

 
    def children(self):
        return [t for (k,t) in self.items() if gdistable(t)]

    def children_names(self):
        return ['.%s'%k for (k,t) in self.items() if gdistable(t)]

    def theano_sampler_helper(self, memo, s_rng):
        for t in self.children():
            t.theano_sampler_helper(memo, s_rng)
        val_dict = dict([(k,get_value(t,memo)) for k,t in self.items()])
        memo[id(self)] = (memo[id(self)],val_dict)

    def nth_theano_sample(self, n, idxdict, valdict):    
        return dict([(k,self.nth_sample(t,n,idxdict,valict)) for (k,t) in self.items()])

    
class gDist(gDict):
        
    def __init__(self,genson_string):
        parser = genson.parser.GENSONParser()
        genson_obj = parser.parse_string(genson_string)
        print('genson obj',genson_obj)
        super(gDist,self).__init__(genson_obj) 
        self.genson_generator = genson.JSONGenerator(genson_obj)
    
    def sample(self,rng,reset = True):
        try:
            return self.genson_generator.next() 
        except StopIteration:
            if reset:
                self.genson_generator.reset()
                return self.genson_generator.next()
            else:
                raise StopIteration
            
            
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
        self.correct_elems(memo,corrected_memo)
        memo = corrected_memo
        self.theano_sampler_helper(memo, s_rng)
        if len(memo) != len(self.flatten()):
            print('memo',memo)
            print('self',self)
            assert (len(memo) == len(self.flatten()))        
        idxs, vals = zip(*[memo[id(n_i)] for n_i in self.flatten()])
        return idxs, vals, s_N
        

class gBinOp(gSON):
    params = ['a','b']

    op_dict = [('+',tensor.add),
               ('-',tensor.sub),
               ('*',tensor.mul),
               ('/',tensor.div_proxy),
               ('**',tensor.pow)]

    def make_contents(self):
        super(gBinOp,self).make_contents()
        self.op = self.op_dict[self.genson_obj.op]
        
    def theano_sampler_helper(self, memo, s_rng):
        for c in self.children():
            child.theano_sampler_helper(memo, s_rng)
        a = get_value(self.a,memo)
        b = get_value(self.b,memo)  
        memo[id(self)] = [memo[id(self)],self.op(a,b)]        
        
    def nth_theano_sample(self, n, idxdict, valdict):
        asample = self.nth_sample(self.a, n, idxdict, valdict)
        bsample = self.nth_sample(self.b, n, idxdict, valdict)
        return self.op(asample,bsample)
        

class gFunc(gSON):
    params = ['args','kwargs']
    
    def make_contents(self):
        super(gBinOp,self).make_contents()
        self.func = getattr(tensor,self.genson_obj.name) 
            
    def theano_sampler_helper(self, memo, s_rng):
        for c in self.children():
            child.theano_sampler_helper(memo, s_rng)
        args = tuple(get_value(self.args,memo))
        kwargs = get_value(self.kwargs,memo)   
        memo[id(self)] = [memo[id(self)],self.func(*args,**kwargs)]        

    def nth_theano_sample(self, n, idxdict, valdict):
        argsample = tuple(self.args.nth_theano_sample(n, idxdict, valdict))
        kwargsample = self.kwargs.nth_theano_sample(n, idxdict, valdict)
        return self.func(*argsample,**kwargsample)
        
        
class gRef(gSON):
    
    params = []
    def make_contents(self):
        self.reference = resolve_scoped_reference(self.genson_obj.scope_list[:],
                                                  self.path[:-1])
        self.propagate_up(self.reference)
        
    def propagate_up(self,reference):
        if isinstance(reference,gSON):
            if hasattr(reference,'referenced_by'):
                reference.referenced_by.append(self)
            else:
                reference.referenced_by = [self]

    def get_elems(self,s_rng,elems,memo):
        if isinstance(self.reference,gSON):    
            memo[id(self)] = memo[id(self.reference)]        
        else:
            memo[id(self)] = None
        
    def theano_sampler_helper(self,memo, s_rng):
        if isinstance(self.reference,gSON): 
            memo[id(self)] = memo[id(self.reference)]
        else:
            memo[id(self)] = (memo[id(self)],self.reference)
        
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

    if( element_to_resolve == 'root'):
        return resolve_scoped_reference(ref, [path[0]])

    if( element_to_resolve == 'parent'):
        return resolve_scoped_reference(ref, path[0:-1])

    if( element_to_resolve == 'this'):
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


                    
class gGauss(gSON):
    params = ['mean','stdev']
    
    def theano_sampler_helper(self, memo, s_rng):
        for c in self.children():
            child.theano_sampler_helper(memo, s_rng)
        mu = get_value(self.mean,memo)
        stdev = get_value(self.stdev,memo)
        sigma = stdev**2
        elems = memo[id(self)]
        vals = s_rng.normal(draw_shape=(elems.shape[0],),mu=mu,sigma=sigma)
        memo[id(self)] = (elems,vals)
                
    def nth_theano_sample(self, n, idxdict, valdict):
        return float(valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]])


class gUniform(gSON):
    params = ['min','max']
    
    def theano_sampler_helper(self, memo, s_rng):
        for child in self.children():
            child.theano_sampler_helper(memo, s_rng)
        low = get_value(self.min,memo)
        high = get_value(self.max,memo)
        elems = memo[id(self)]
        vals = s_rng.uniform(draw_shape=(elems.shape[0],),low=low,high=high)
        memo[id(self)] = (elems,vals)
        
    def nth_theano_sample(self, n, idxdict, valdict):
        return float(valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]])


class gChoice(gSON):
    params = ['vals']
    
    def get_elems(self, s_rng, elems, memo):
        n_options = len(self.vals)
        #print 'n_options', n_options
        casevar = s_rng.categorical(
                    p=[1.0 / n_options] * n_options,
                    draw_shape=(elems.shape[0],))
        memo[id(self)] = elems
        for i, child in enumerate(self.vals):
            elems_i = elems[MT.for_theano.where(tensor.eq(i, casevar))]
            child.get_elems(s_rng,elems_i,memo)
            
    def theano_sampler_helper(self, memo, s_rng):
        for c in self.children():
            child.theano_sampler_helper(memo, s_rng)
        n_options = len(self.vals['elements'])
        elems = memo[id(self)]
        casevar = s_rng.categorical(
                    p=[1.0 / n_options] * n_options,
                    draw_shape=(elems.shape[0],))
        memo[id(self)] = (elems,casevar)           

    def nth_theano_sample(self, n, idxdict, valdict):
        case = int(valdict[id(self)][numpy.where(idxdict[id(self)]==n)[0][0]])
        if self.vals[case] in self.children():
            return self.vals[case].nth_theano_sample(n, idxdict, valdict)
        else:
            return self.vals[case]
         
def get_value(x,memo):
    if isinstance(x,gSON):
        return memo[id(x)][1]
    else:
        return x

class GensonBandit(base.Bandit):
    def __init__(self,genson_file):
        template = gDist(open(genson_file).read())
        base.Bandit.__init__(self.template)
    
