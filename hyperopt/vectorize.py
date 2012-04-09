import sys

import numpy as np

from pyll import Apply
from pyll import as_apply
from pyll import dfs
from pyll import scope
from pyll import stochastic
from pyll import clone_merge

stoch = stochastic.implicit_stochastic_symbols


def ERR(msg):
    print >> sys.stderr, msg

@scope.define_pure
def vchoice_split(idxs, choices, n_options):
    rval = [[] for ii in range(n_options)]
    if len(idxs) != len(choices):
        raise ValueError('idxs and choices different len',
                (len(idxs), len(choices)))
    for ii, cc in zip(idxs, choices):
        rval[cc].append(ii)
    return rval


@scope.define_pure
def vchoice_merge(idxs, choices, *vals):
    rval = []
    assert len(idxs) == len(choices)
    for idx, ch in zip(idxs, choices):
        vi, vv = vals[ch]
        rval.append(vv[list(vi).index(idx)])
    return rval


@scope.define_pure
def idxs_map(idxs, cmd, *args, **kwargs):
    """
    Return the cmd applied at positions idxs, by retrieving args and kwargs
    from the (idxs, vals) pair elements of `args` and `kwargs`.

    N.B. args and kwargs may generally include information for more idx values
    than are requested by idxs.
    """
    # XXX: consider insisting on sorted idxs
    # XXX: use np.searchsorted instead of dct

    if 0: # these should all be true, but evaluating them is slow
        for ii, (idxs_ii, vals_ii) in enumerate(args):
            for jj in idxs: assert jj in idxs_ii
        for kw, (idxs_kw, vals_kw) in kwargs.items():
            for jj in idxs: assert jj in idxs_kw

    args_imap = []
    for idxs_j, vals_j in args:
        if len(idxs_j):
            args_imap.append(dict(zip(idxs_j, vals_j)))
        else:
            args_imap.append({})

    kwargs_imap = {}
    for kw, (idxs_j, vals_j) in kwargs.items():
        if len(idxs_j):
            kwargs_imap[kw] = dict(zip(idxs_j, vals_j))
        else:
            kwargs_imap[kw] = {}

    f = scope._impls[cmd]
    rval = []
    for ii in idxs:
        try:
            args_nn = [arg_imap[ii] for arg_imap in args_imap]
        except:
            ERR('args_nn %s' % cmd)
            ERR('ii %s' % ii)
            ERR('arg_imap %s' % str(arg_imap))
            ERR('args_imap %s' % str(args_imap))
            raise
        try:
            kwargs_nn = dict([(kw, arg_imap[ii])
                for kw, arg_imap in kwargs_imap.items()])
        except:
            ERR('args_nn %s' % cmd)
            ERR('ii %s' % ii)
            ERR('kw %s' % kw)
            ERR('arg_imap %s' % str(arg_imap))
            raise
        try:
            rval_nn = f(*args_nn, **kwargs_nn)
        except:
            ERR('error calling impl of %s' % cmd)
            raise
        rval.append(rval_nn)
    return rval


@scope.define_pure
def idxs_take(idxs, vals, which):
    """
    Return `vals[which]` where `which` is a subset of `idxs`
    """
    # XXX: consider insisting on sorted idxs
    # XXX: use np.searchsorted instead of dct
    assert len(idxs) == len(vals)
    table = dict(zip(idxs, vals))
    return [table[w] for w in which]


@scope.define_pure
def uniq(lst):
    s = set()
    rval = []
    for l in lst:
        if id(l) not in s:
            s.add(id(l))
            rval.append(l)
    return rval


def vectorize_stochastic(orig):
    if orig.name == 'idxs_map' and orig.pos_args[1]._obj in stoch:
        # -- this is an idxs_map of a random draw of distribution `dist`
        idxs = orig.pos_args[0]
        dist = orig.pos_args[1]._obj
        def foo(arg):
            # -- each argument is an idxs, vals pair
            assert arg.name == 'pos_args'
            assert len(arg.pos_args) == 2
            arg_vals = arg.pos_args[1]
            if (arg_vals.name == 'asarray'
                    and arg_vals.inputs()[0].name == 'repeat'):
                # -- draws are iid, so forget about
                #    repeating the distribution parameters
                repeated_thing = arg_vals.inputs()[0].inputs()[1]
                return repeated_thing
            else:
                if arg.pos_args[0] is idxs:
                    return arg_vals
                else:
                    # -- arg.pos_args[0] is a superset of idxs
                    #    TODO: slice out correct elements using
                    #    idxs_take, but more importantly - test this case.
                    raise NotImplementedError()
        new_pos_args = [foo(arg) for arg in orig.pos_args[2:]]
        new_named_args = [[aname, foo(arg)]
                for aname, arg in orig.named_args]
        vnode = Apply(dist, new_pos_args, new_named_args, None)
        n_times = scope.len(idxs)
        if 'size' in dict(vnode.named_args):
            raise NotImplementedError('random node already has size')
        vnode.named_args.append(['size', n_times])
        return vnode
    else:
        return orig


def replace_repeat_stochastic(expr, return_memo=False):
    nodes = dfs(expr)
    memo = {}
    for ii, orig in enumerate(nodes):
        if orig.name == 'idxs_map' and orig.pos_args[1]._obj in stoch:
            # -- this is an idxs_map of a random draw of distribution `dist`
            idxs = orig.pos_args[0]
            dist = orig.pos_args[1]._obj
            def foo(arg):
                # -- each argument is an idxs, vals pair
                assert arg.name == 'pos_args'
                assert len(arg.pos_args) == 2
                arg_vals = arg.pos_args[1]
                if (arg_vals.name == 'asarray'
                        and arg_vals.inputs()[0].name == 'repeat'):
                    # -- draws are iid, so forget about
                    #    repeating the distribution parameters
                    repeated_thing = arg_vals.inputs()[0].inputs()[1]
                    return repeated_thing
                else:
                    if arg.pos_args[0] is idxs:
                        return arg_vals
                    else:
                        # -- arg.pos_args[0] is a superset of idxs
                        #    TODO: slice out correct elements using
                        #    idxs_take, but more importantly - test this case.
                        raise NotImplementedError()
            new_pos_args = [foo(arg) for arg in orig.pos_args[2:]]
            new_named_args = [[aname, foo(arg)]
                    for aname, arg in orig.named_args]
            vnode = Apply(dist, new_pos_args, new_named_args, None)
            n_times = scope.len(idxs)
            if 'size' in dict(vnode.named_args):
                raise NotImplementedError('random node already has size')
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
    Convert a pyll expression representing a single trial into a pyll
    expression representing multiple trials.

    The resulting multi-trial expression is not meant to be evaluated
    directly. It is meant to serve as the input to a BanditAlgo.

    """

    def __init__(self, expr, expr_idxs, build=True, rseed=1):
        self.expr = expr
        self.expr_idxs = expr_idxs
        self._idxs_memo = {}  # -- map node -> [idxs, idxs, ...]
        self._vals_memo = {}  # -- map node -> [vals, vals, ...]
        self.dfs_nodes = dfs(expr)
        self.node_id = {}
        self.param_nodes = {}
        for ii, node in enumerate(self.dfs_nodes):
            self.node_id[node] = 'node_%i' % ii
            if node.name == 'hyperopt_param':
                label = node.arg['label'].obj
                self.node_id[node.arg['obj']] = label
                self.param_nodes[label] = node.arg['obj']
        rng = np.random.RandomState(rseed)

        # -- recursive construction
        #    This makes one term in each idxs, vals memo for every
        #    directed path through the switches in the graph.
        self.build_idxs_vals(expr, expr_idxs)

        if 0:
          for key in self._idxs_memo:
            print '=' * 80
            print 'VH._idxs_memo'
            print id(key)
            print ' key'
            print key
            for ii, (i, v) in enumerate(
                    zip(self._idxs_memo[key], self._vals_memo[key])):
                print ' _idxs_memo', ii
                print i
                print ' _vals_memo', ii
                print v

        assert set(self._idxs_memo.keys()) < set(self.dfs_nodes)

        self.idxs_memo = {}
        self.vals_memo = {}
        assert len(self._idxs_memo[expr]) == 1
        # -- consolidate the idxs and vals so that each node is only evaluated
        # once

        for node in dfs(self._vals_memo[expr][0]):
            if len(self._idxs_memo.get(node, [])) > 1:
                # -- merge together these two samples
                idxs_list = self._idxs_memo[node]

                if len(set(idxs_list)) > 1:
                    u_idxs = scope.array_union1(scope.uniq(idxs_list))
                else:
                    u_idxs = idxs_list[0]

                vals_list = self._vals_memo[node]

                if node.name == 'switch':
                    choice = node.pos_args[0]
                    options = node.pos_args[1:]

                    u_choices = scope.idxs_take(
                            self.idxs_memo[choice],
                            self.vals_memo[choice],
                            u_idxs)

                    args_idxs = scope.vchoice_split(u_idxs, u_choices,
                            len(options))
                    u_vals = scope.vchoice_merge(u_idxs, u_choices)
                    for opt_ii, idxs_ii in zip(options, args_idxs):
                        u_vals.pos_args.append(as_apply([
                            self.idxs_memo[opt_ii], self.vals_memo[opt_ii]]))

                    print 'finalizing switch', u_vals

                elif node.name in stoch:
                    # -- this case is separate because we're going to change
                    # the program semantics. If multiple stochastic nodes
                    # are being merged, it means just sample once, and then
                    # index multiple subsets
                    print 'finalizing', node.name
                    assert all(thing.name == node.name for thing in vals_list)

                    # -- assert that all the args except size to each
                    # function are the same
                    vv0 = vals_list[0]
                    vv0d = dict(vv0.arg, rng=None, size=None)
                    for vv in vals_list[1:]:
                        assert vv0d == dict(vv.arg, rng=None, size=None)

                    u_vals = vals_list[0].clone_from_inputs(
                            vals_list[0].inputs())
                    u_vals.set_kwarg('size', scope.len(u_idxs))
                    u_vals.set_kwarg('rng',
                            as_apply(
                                np.random.RandomState(
                                    rng.randint(int(2**30)))))

                else:
                    print 'creating idxs map', node.name
                    u_vals = scope.idxs_map(u_idxs, node.name)
                    u_vals.pos_args.extend(node.pos_args)
                    u_vals.named_args.extend(node.named_args)
                    for arg in node.inputs():
                        u_vals.replace_input(arg,
                                as_apply([
                                    self.idxs_memo[arg],
                                    self.vals_memo[arg]]))

            else:
                print '=' * 80
                print node

            self.idxs_memo[node] = u_idxs
            self.vals_memo[node] = u_vals


        del self._idxs_memo
        del self._vals_memo

        if 0:
            print as_apply([
                        self.idxs_memo[expr],
                        self.vals_memo[expr]])

        merge_memo = {}
        m_idxs_vals = clone_merge(
                as_apply([
                    self.idxs_memo[expr],
                    self.vals_memo[expr]]),
                memo=merge_memo,
                merge_literals=True)

        for key in self.idxs_memo:
            k_i = self.idxs_memo[key]
            k_v = self.vals_memo[key]
            if k_i in merge_memo:
                self.idxs_memo[key] = merge_memo[k_i]
            if k_v in merge_memo:
                self.vals_memo[key] = merge_memo[k_v]

    def build_idxs_vals(self, node, idxs):
        """
        This recursive procedure should be called on an output-node.
        """
        if node.name == 'hyperopt_param':
            return self.build_idxs_vals(node.arg['obj'], idxs)

        if node.name == 'hyperopt_result':
            return self.build_idxs_vals(node.arg['obj'], idxs)

        self._idxs_memo.setdefault(node, []).append(idxs)

        if node.name == 'literal':
            n_times = scope.len(idxs)
            vnode = scope.asarray(scope.repeat(n_times, node))

        elif node.name == 'switch':
            choice = node.pos_args[0]
            choices = self.build_idxs_vals(choice, idxs)

            options = node.pos_args[1:]
            n_options = len(options)
            args_idxs = scope.vchoice_split(idxs, choices, n_options)
            vnode = scope.vchoice_merge(idxs, choices)
            for opt_ii, idxs_ii in zip(options, args_idxs):
                opt_vnode = self.build_idxs_vals(opt_ii, idxs_ii)
                vnode.pos_args.append(as_apply([idxs_ii, opt_vnode]))

        else:
            vnode = scope.idxs_map(idxs, node.name)
            vnode.pos_args.extend(node.pos_args)
            vnode.named_args.extend(node.named_args)
            for arg in node.inputs():
                arg_vnode = self.build_idxs_vals(arg, idxs)
                vnode.replace_input(arg, as_apply([idxs, arg_vnode]))
            vnode = vectorize_stochastic(vnode)

        self._vals_memo.setdefault(node, []).append(vnode)
        assert len(self._vals_memo[node]) == len(self._idxs_memo[node])
        return vnode

    def idxs_by_id(self):
        rval = dict([(self.node_id[node], idxs)
            for node, idxs in self.idxs_memo.items()])
        return rval

    def vals_by_id(self):
        rval = dict([(self.node_id[node], vals)
            for node, vals in self.vals_memo.items()])
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



if 0:
        for node in dfs(self._vals_memo[expr][0]):
            if node.name == 'literal':
                u_idxs = expr_idxs
                u_vals = scope.asarray(
                        scope.repeat(
                            scope.len(u_idxs),
                            node))
            elif node.name in ('hyperopt_param', 'hyperopt_result'):
                u_idxs = self.idxs_memo[node.arg['obj']]
                u_vals = self.vals_memo[node.arg['obj']]

            elif node in self._idxs_memo:
                idxs_list = self._idxs_memo[node]

                if len(set(idxs_list)) > 1:
                    u_idxs = scope.array_union1(scope.uniq(idxs_list))
                else:
                    u_idxs = idxs_list[0]

                vals_list = self._vals_memo[node]

                if node.name == 'switch':
                    choice = node.pos_args[0]
                    options = node.pos_args[1:]

                    u_choices = scope.idxs_take(
                            self.idxs_memo[choice],
                            self.vals_memo[choice],
                            u_idxs)

                    args_idxs = scope.vchoice_split(u_idxs, u_choices,
                            len(options))
                    u_vals = scope.vchoice_merge(u_idxs, u_choices)
                    for opt_ii, idxs_ii in zip(options, args_idxs):
                        u_vals.pos_args.append(as_apply([
                            self.idxs_memo[opt_ii], self.vals_memo[opt_ii]]))

                    print 'finalizing switch', u_vals

                elif node.name in stoch:
                    # -- this case is separate because we're going to change
                    # the program semantics. If multiple stochastic nodes
                    # are being merged, it means just sample once, and then
                    # index multiple subsets
                    print 'finalizing', node.name
                    assert all(thing.name == node.name for thing in vals_list)

                    # -- assert that all the args except size to each
                    # function are the same
                    vv0 = vals_list[0]
                    vv0d = dict(vv0.arg, rng=None, size=None)
                    for vv in vals_list[1:]:
                        assert vv0d == dict(vv.arg, rng=None, size=None)

                    u_vals = vals_list[0].clone_from_inputs(
                            vals_list[0].inputs())
                    u_vals.set_kwarg('size', scope.len(u_idxs))
                    u_vals.set_kwarg('rng',
                            as_apply(
                                np.random.RandomState(
                                    rng.randint(int(2**30)))))

                else:
                    print 'creating idxs map', node.name
                    u_vals = scope.idxs_map(u_idxs, node.name)
                    u_vals.pos_args.extend(node.pos_args)
                    u_vals.named_args.extend(node.named_args)
                    for arg in node.inputs():
                        u_vals.replace_input(arg,
                                as_apply([
                                    self.idxs_memo[arg],
                                    self.vals_memo[arg]]))

            else:
                print '=' * 80
                print node

            self.idxs_memo[node] = u_idxs
            self.vals_memo[node] = u_vals
