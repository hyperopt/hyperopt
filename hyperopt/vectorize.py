import sys

from pyll import Apply
from pyll import as_apply
from pyll import dfs
from pyll import scope
from pyll import stochastic

def ERR(msg):
    print >> sys.stderr, msg

@scope.define
def vchoice_split(idxs, choices, n_options):
    rval = [[] for ii in range(n_options)]
    if len(idxs) != len(choices):
        raise ValueError('idxs and choices different len',
                (len(idxs), len(choices)))
    for ii, cc in zip(idxs, choices):
        rval[cc].append(ii)
    return rval


@scope.define
def vchoice_merge(idxs, choices, *vals):
    rval = []
    assert len(idxs) == len(choices)
    for idx, ch in zip(idxs, choices):
        vi, vv = vals[ch]
        rval.append(vv[list(vi).index(idx)])
    return rval


@scope.define
def idxs_map(idxs, cmd, *args, **kwargs):
    for ii, (idxs_ii, vals_ii) in enumerate(args):
        for jj in idxs: assert jj in idxs_ii
    for kw, (idxs_kw, vals_kw) in kwargs.items():
        for jj in idxs: assert jj in idxs_kw
    f = scope._impls[cmd]
    rval = []
    for ii in idxs:
        try:
            args_nn = [vals_j[list(idxs_j).index(ii)] for (idxs_j, vals_j) in args]
        except:
            ERR('args_nn %s' % cmd)
            ERR('ii %s' % ii)
            ERR('idxs %s' % str(idxs))
            ERR('idxs_j %s' % str(idxs_j))
            ERR('vals_j %s' % str(vals_j))
            raise
        try:
            kwargs_nn = dict([(kw, vals_j[list(idxs_j).index(ii)])
                for kw, (idxs_j, vals_j) in kwargs.items()])
        except:
            ERR('args_nn %s' % cmd)
            ERR('ii %s' % ii)
            ERR('kw %s' % kw)
            ERR('idxs %s' % str(idxs))
            ERR('idxs_j %s' % str(idxs_j))
            ERR('vals_j %s' % str(vals_j))
            raise
        try:
            rval_nn = f(*args_nn, **kwargs_nn)
        except:
            ERR('error calling impl of %s' % cmd)
            raise
        rval.append(rval_nn)
    return rval


def replace_repeat_stochastic(expr, return_memo=False):
    stoch = stochastic.implicit_stochastic_symbols
    nodes = dfs(expr)
    memo = {}
    for ii, orig in enumerate(nodes):
        if orig.name == 'idxs_map' and orig.pos_args[1]._obj in stoch:
            # -- this is an idxs_map of a random draw of distribution `dist`
            idxs = orig.pos_args[0]
            dist = orig.pos_args[1]._obj
            inputs = []
            for arg in orig.pos_args[2:]:
                # -- each argument is an idxs, vals pair
                assert arg.name == 'pos_args'
                assert len(arg.pos_args) == 2
                assert arg.pos_args[0] is idxs, str(orig)
                arg_vals = arg.pos_args[1]
                if (arg_vals.name == 'asarray'
                        and arg_vals.inputs()[0].name == 'repeat'):
                    # -- draws are iid, so forget about
                    #    repeating the distribution parameters
                    repeated_thing = arg_vals.inputs()[0].inputs()[1]
                    inputs.append(repeated_thing)
                else:
                    inputs.append(arg_vals)
            if orig.named_args:
                raise NotImplementedError()
            vnode = Apply(dist, inputs, orig.named_args, None)
            n_times = scope.len(idxs)
            vnode.named_args.append(['size', n_times])
            # -- loop over all nodes that *use* this one, and change them
            for client in nodes[ii+1:]:
                client.replace_input(orig, vnode)
            if expr is orig:
                expr = vnode
            memo[orig] = vnode
    if return_memo:
        return expr, memo
    else:
        return expr



class VectorizeHelper(object):
    """
    Example:
        u0 = uniform(1, 2)
        u1 = uniform(2, 3)
        c = one_of(u0, u1)
        expr = {'u1': u1, 'c': c}
    becomes
        N
        expr_idxs = range(N)
        choices = randint(2, len(expr_idxs))
        c0_idxs, c1_idxs = vchoice_split(expr_idxs, choices)
        c0_vals = vdraw(len(c0_idxs), 'uniform', 1, 2)
        c1_vals = vdraw(len(c1_idxs), 'uniform', 2, 3)
    """
    def __init__(self, expr, expr_idxs):
        self.expr = expr
        self.expr_idxs = expr_idxs
        self.idxs_memo = {expr: expr_idxs}
        self.vals_memo = {}
        self.choice_memo = {}
        self.dfs_nodes = dfs(expr)
        self.node_id = dict([(node, 'node_%i' % ii)
            for ii, node in enumerate(dfs(expr))])

    def merge(self, idxs, node):
        if node in self.idxs_memo:
            other = self.idxs_memo[node]
            if other is not idxs:
                self.idxs_memo[node] = scope.array_union(idxs, self.idxs_memo[node])
        else:
            self.idxs_memo[node] = idxs

    # -- separate method for testing
    def build_idxs(self):
        for node in reversed(self.dfs_nodes):
            node_idxs = self.idxs_memo[node]
            if node.name == 'one_of':
                n_options  = len(node.pos_args)
                choices = scope.randint(n_options, size=scope.len(node_idxs))
                self.choice_memo[node] = choices
                self.merge(node_idxs, choices)
                self.node_id[choices] = 'node_%i' % len(self.node_id)
                sub_idxs = scope.vchoice_split(node_idxs, choices, n_options)
                for ii, arg in enumerate(node.pos_args):
                    self.merge(sub_idxs[ii], arg)
            else:
                for arg in node.inputs():
                    self.merge(node_idxs, arg)

    # -- separate method for testing
    def build_vals(self):
        for node in self.dfs_nodes:
            if node.name == 'literal':
                n_times = scope.len(self.idxs_memo[node])
                vnode = scope.asarray(scope.repeat(n_times, node))
            elif node in self.choice_memo:
                # -- choices are natively vectorized
                choices = self.choice_memo[node]
                self.vals_memo[choices] = choices
                # -- this stitches together the various sub-graphs
                #    to define the original node
                vnode = scope.vchoice_merge(
                        self.idxs_memo[node],
                        self.choice_memo[node])
                vnode.pos_args.extend([
                    as_apply([
                        self.idxs_memo[inode],
                        self.vals_memo[inode]])
                    for inode in node.pos_args])
            else:
                vnode = scope.idxs_map(self.idxs_memo[node], node.name)
                vnode.pos_args.extend(node.pos_args)
                vnode.named_args.extend(node.named_args)
                for arg in node.inputs():
                    vnode.replace_input(arg,
                            as_apply([
                                self.idxs_memo[arg],
                                self.vals_memo[arg]]))
            self.vals_memo[node] = vnode

    def idxs_by_id(self):
        rval = dict([(self.node_id[node], idxs)
            for node, idxs in self.idxs_memo.items()])
        return rval

    def vals_by_id(self):
        rval = dict([(self.node_id[node], idxs)
            for node, idxs in self.vals_memo.items()])
        return rval

    def name_by_id(self):
        rval = dict([(nid, node.name)
            for (node, nid) in self.node_id.items()])
        return rval

    def pretty_by_id(self):
        names = node_names(self.expr)
        rval = dict([(nid, names.get(node, 'missing'))
            for (node, nid) in self.node_id.items()])
        return rval


def pretty_names_helper(expr, seq, seqset, prefixes, names):
    if expr in seqset:
        return
    assert isinstance(expr, Apply)
    seqset.add(expr)
    if expr.name == 'dict':
        for ii, (aname, aval) in enumerate(expr.named_args):
            pretty_names_helper(aval, seq, seqset,
                    prefixes + (('%s' % aname),),
                    names)
    else:
        for ii, aval in enumerate(expr.pos_args):
            pretty_names_helper(aval, seq, seqset,
                    prefixes + ('arg:%i' % (ii,),),
                    names)
        for ii, (aname, aval) in enumerate(expr.named_args):
            pretty_names_helper(aval, seq, seqset,
                    prefixes + ('kw:%s' % (aname,),),
                    names)
    names.append('.'.join(prefixes))
    seq.append(expr)


def pretty_names(expr, prefix=None):
    dfs_order = dfs(expr)
    # -- compute the seq like pyll.dfs just to ensure that
    #    the order of our names matches the dfs order.
    #    It's not clear to me right now that the match is important,
    #    but it's certainly suspicious if not.
    seq = []
    names = []
    seqset = set()
    if prefix is None:
        prefixes = ()
    else:
        prefixes = prefix,
    pretty_names_helper(expr, seq, seqset, prefixes, names)
    assert seq == dfs_order
    return dict(zip(seq, names))

