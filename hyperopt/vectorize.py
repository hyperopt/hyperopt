import sys

import numpy as np

from .pyll import Apply
from .pyll import as_apply
from .pyll import dfs
from .pyll import toposort
from .pyll import scope
from .pyll import stochastic

stoch = stochastic.implicit_stochastic_symbols


def ERR(msg):
    print("hyperopt.vectorize.ERR", msg, file=sys.stderr)


@scope.define_pure
def vchoice_split(idxs, choices, n_options):
    rval = [[] for ii in range(n_options)]
    if len(idxs) != len(choices):
        raise ValueError("idxs and choices different len", (len(idxs), len(choices)))
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

    if 0:  # these should all be true, but evaluating them is slow
        for ii, (idxs_ii, vals_ii) in enumerate(args):
            for jj in idxs:
                assert jj in idxs_ii
        for kw, (idxs_kw, vals_kw) in list(kwargs.items()):
            for jj in idxs:
                assert jj in idxs_kw

    args_imap = []
    for idxs_j, vals_j in args:
        d = dict(list(zip(idxs_j, vals_j))) if len(idxs_j) else {}
        args_imap.append(d)

    kwargs_imap = {}
    for kw, (idxs_j, vals_j) in list(kwargs.items()):
        if len(idxs_j):
            kwargs_imap[kw] = dict(list(zip(idxs_j, vals_j)))
        else:
            kwargs_imap[kw] = {}

    f = scope._impls[cmd]
    rval = []
    for ii in idxs:
        try:
            args_nn = [arg_imap[ii] for arg_imap in args_imap]
        except:
            ERR("args_nn %s" % cmd)
            ERR("ii %s" % ii)
            ERR("arg_imap %s" % str(args_imap))
            ERR("args_imap %s" % str(args_imap))
            raise
        try:
            kwargs_nn = {kw: arg_imap[ii] for kw, arg_imap in list(kwargs_imap.items())}
        except:
            ERR("args_nn %s" % cmd)
            ERR("ii %s" % ii)
            ERR("kw %s" % kw)
            ERR("arg_imap %s" % str(args_imap))
            raise
        try:
            rval_nn = f(*args_nn, **kwargs_nn)
        except:
            ERR("error calling impl of %s" % cmd)
            raise
        rval.append(rval_nn)
    return rval


@scope.define_pure
def idxs_take(idxs, vals, which):
    """
    Return `vals[which]` where `which` is a subset of `idxs`
    """
    # TODO: consider insisting on sorted idxs
    # TODO: use np.searchsorted instead of dct
    assert len(idxs) == len(vals)
    table = dict(list(zip(idxs, vals)))
    return np.asarray([table[w] for w in which])


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
    if orig.name == "idxs_map" and orig.pos_args[1]._obj in stoch:
        # -- this is an idxs_map of a random draw of distribution `dist`
        idxs = orig.pos_args[0]
        dist = orig.pos_args[1]._obj

        def foo(arg):
            # -- each argument is an idxs, vals pair
            assert arg.name == "pos_args"
            assert len(arg.pos_args) == 2
            arg_vals = arg.pos_args[1]

            # XXX: write a pattern-substitution rule for this case
            if arg_vals.name == "idxs_take":
                if arg_vals.arg["vals"].name == "asarray":
                    if arg_vals.arg["vals"].inputs()[0].name == "repeat":
                        # -- draws are iid, so forget about
                        #    repeating the distribution parameters
                        repeated_thing = arg_vals.arg["vals"].inputs()[0].inputs()[1]
                        return repeated_thing
            if arg.pos_args[0] is idxs:
                return arg_vals
            else:
                # -- arg.pos_args[0] is a superset of idxs
                #    TODO: slice out correct elements using
                #    idxs_take, but more importantly - test this case.
                raise NotImplementedError()

        new_pos_args = [foo(arg) for arg in orig.pos_args[2:]]
        new_named_args = [[aname, foo(arg)] for aname, arg in orig.named_args]
        vnode = Apply(dist, new_pos_args, new_named_args, o_len=None)
        n_times = scope.len(idxs)
        if "size" in dict(vnode.named_args):
            raise NotImplementedError("random node already has size")
        vnode.named_args.append(["size", n_times])
        return vnode
    else:
        return orig


def replace_repeat_stochastic(expr, return_memo=False):
    nodes = dfs(expr)
    memo = {}
    for ii, orig in enumerate(nodes):
        if orig.name == "idxs_map" and orig.pos_args[1]._obj in stoch:
            # -- this is an idxs_map of a random draw of distribution `dist`
            idxs = orig.pos_args[0]
            dist = orig.pos_args[1]._obj

            def foo(arg):
                # -- each argument is an idxs, vals pair
                assert arg.name == "pos_args"
                assert len(arg.pos_args) == 2
                arg_vals = arg.pos_args[1]
                if arg_vals.name == "asarray" and arg_vals.inputs()[0].name == "repeat":
                    # -- draws are iid, so forget about
                    #    repeating the distribution parameters
                    repeated_thing = arg_vals.inputs()[0].inputs()[1]
                    return repeated_thing
                else:
                    if arg.pos_args[0] is idxs:
                        return arg_vals
                    # -- arg.pos_args[0] is a superset of idxs
                    #    TODO: slice out correct elements using
                    #    idxs_take, but more importantly - test this case.
                    raise NotImplementedError()

            new_pos_args = [foo(arg) for arg in orig.pos_args[2:]]
            new_named_args = [[aname, foo(arg)] for aname, arg in orig.named_args]
            vnode = Apply(dist, new_pos_args, new_named_args, None)
            n_times = scope.len(idxs)
            if "size" in dict(vnode.named_args):
                raise NotImplementedError("random node already has size")
            vnode.named_args.append(["size", n_times])
            # -- loop over all nodes that *use* this one, and change them
            for client in nodes[ii + 1 :]:
                client.replace_input(orig, vnode)
            if expr is orig:
                expr = vnode
            memo[orig] = vnode
    if return_memo:
        return expr, memo
    return expr


class VectorizeHelper:
    """
    Convert a pyll expression representing a single trial into a pyll
    expression representing multiple trials.

    The resulting multi-trial expression is not meant to be evaluated
    directly. It is meant to serve as the input to a suggest algo.

    idxs_memo - node in expr graph -> all elements we might need for it
    take_memo - node in expr graph -> all exprs retrieving computed elements

    """

    def __init__(self, expr, expr_idxs, build=True):
        self.expr = expr
        self.expr_idxs = expr_idxs
        self.dfs_nodes = dfs(expr)
        self.params = {}
        for ii, node in enumerate(self.dfs_nodes):
            if node.name == "hyperopt_param":
                label = node.arg["label"].obj
                self.params[label] = node.arg["obj"]
        # -- recursive construction
        #    This makes one term in each idxs, vals memo for every
        #    directed path through the switches in the graph.

        self.idxs_memo = {}  # node -> union, all idxs computed
        self.take_memo = {}  # node -> list of idxs_take retrieving node vals
        self.v_expr = self.build_idxs_vals(expr, expr_idxs)

        # TODO: graph-optimization pass to remove cruft:
        #  - unions of 1
        #  - unions of full sets with their subsets
        #  - idxs_take that can be merged

        self.assert_integrity_idxs_take()

    def assert_integrity_idxs_take(self):
        idxs_memo = self.idxs_memo
        take_memo = self.take_memo
        after = dfs(self.expr)
        assert after == self.dfs_nodes
        assert set(idxs_memo.keys()) == set(take_memo.keys())
        for node in idxs_memo:
            idxs = idxs_memo[node]
            assert idxs.name == "array_union"
            vals = take_memo[node][0].pos_args[1]
            for take in take_memo[node]:
                assert take.name == "idxs_take"
                assert [idxs, vals] == take.pos_args[:2]

    def build_idxs_vals(self, node, wanted_idxs):
        """
        This recursive procedure should be called on an output-node.
        """
        checkpoint_asserts = False

        def checkpoint():
            if checkpoint_asserts:
                self.assert_integrity_idxs_take()
                if node in self.idxs_memo:
                    toposort(self.idxs_memo[node])
                if node in self.take_memo:
                    for take in self.take_memo[node]:
                        toposort(take)

        checkpoint()

        # wanted_idxs are fixed, whereas idxs_memo
        # is full of unions, that can grow in subsequent recursive
        # calls to build_idxs_vals with node as argument.
        assert wanted_idxs != self.idxs_memo.get(node)

        # -- easy exit case
        if node.name == "hyperopt_param":
            # -- ignore, not vectorizing
            return self.build_idxs_vals(node.arg["obj"], wanted_idxs)

        # -- easy exit case
        elif node.name == "hyperopt_result":
            # -- ignore, not vectorizing
            return self.build_idxs_vals(node.arg["obj"], wanted_idxs)

        # -- literal case: always take from universal set
        elif node.name == "literal":
            if node in self.idxs_memo:
                all_idxs, all_vals = self.take_memo[node][0].pos_args[:2]
                wanted_vals = scope.idxs_take(all_idxs, all_vals, wanted_idxs)
                self.take_memo[node].append(wanted_vals)
                checkpoint()
            else:
                # -- initialize idxs_memo to full set
                all_idxs = self.expr_idxs
                n_times = scope.len(all_idxs)
                # -- put array_union into graph for consistency, though it is
                # not necessary
                all_idxs = scope.array_union(all_idxs)
                self.idxs_memo[node] = all_idxs
                all_vals = scope.asarray(scope.repeat(n_times, node))
                wanted_vals = scope.idxs_take(all_idxs, all_vals, wanted_idxs)
                assert node not in self.take_memo
                self.take_memo[node] = [wanted_vals]
                checkpoint()
            return wanted_vals

        # -- switch case: complicated
        elif node.name == "switch":
            if node in self.idxs_memo and wanted_idxs in self.idxs_memo[node].pos_args:
                # -- phew, easy case
                all_idxs, all_vals = self.take_memo[node][0].pos_args[:2]
                wanted_vals = scope.idxs_take(all_idxs, all_vals, wanted_idxs)
                self.take_memo[node].append(wanted_vals)
                checkpoint()
            else:
                # -- we need to add some indexes
                if node in self.idxs_memo:
                    all_idxs = self.idxs_memo[node]
                    assert all_idxs.name == "array_union"
                    all_idxs.pos_args.append(wanted_idxs)
                else:
                    all_idxs = scope.array_union(wanted_idxs)

                choice = node.pos_args[0]
                all_choices = self.build_idxs_vals(choice, all_idxs)

                options = node.pos_args[1:]
                args_idxs = scope.vchoice_split(all_idxs, all_choices, len(options))
                all_vals = scope.vchoice_merge(all_idxs, all_choices)
                for opt_ii, idxs_ii in zip(options, args_idxs):
                    all_vals.pos_args.append(
                        as_apply([idxs_ii, self.build_idxs_vals(opt_ii, idxs_ii)])
                    )

                wanted_vals = scope.idxs_take(
                    all_idxs,  # -- may grow in future
                    all_vals,  # -- may be replaced in future
                    wanted_idxs,
                )  # -- fixed.
                if node in self.idxs_memo:
                    assert self.idxs_memo[node].name == "array_union"
                    self.idxs_memo[node].pos_args.append(wanted_idxs)
                    for take in self.take_memo[node]:
                        assert take.name == "idxs_take"
                        take.pos_args[1] = all_vals
                    self.take_memo[node].append(wanted_vals)
                else:
                    self.idxs_memo[node] = all_idxs
                    self.take_memo[node] = [wanted_vals]
                checkpoint()

        # -- general case
        else:
            # -- this is a general node.
            #    It is generally handled with idxs_memo,
            #    but vectorize_stochastic may immediately transform it into
            #    a more compact form.
            if node in self.idxs_memo and wanted_idxs in self.idxs_memo[node].pos_args:
                # -- phew, easy case
                for take in self.take_memo[node]:
                    if take.pos_args[2] == wanted_idxs:
                        return take
                raise NotImplementedError("how did this happen?")
                # all_idxs, all_vals = self.take_memo[node][0].pos_args[:2]
                # wanted_vals = scope.idxs_take(all_idxs, all_vals, wanted_idxs)
                # self.take_memo[node].append(wanted_vals)
                # checkpoint()
            else:
                # XXX
                # -- determine if wanted_idxs is actually a subset of the idxs
                # that we are already computing.  This is not only an
                # optimization, but prevents the creation of cycles, which
                # would otherwise occur if we have a graph of the form
                # switch(f(a), g(a), 0). If there are other switches inside f
                # and g, does this get trickier?

                # -- assume we need to add some indexes
                checkpoint()
                if node in self.idxs_memo:
                    all_idxs = self.idxs_memo[node]

                else:
                    all_idxs = scope.array_union(wanted_idxs)
                checkpoint()

                all_vals = scope.idxs_map(all_idxs, node.name)
                for ii, aa in enumerate(node.pos_args):
                    all_vals.pos_args.append(
                        as_apply([all_idxs, self.build_idxs_vals(aa, all_idxs)])
                    )
                    checkpoint()
                for ii, (nn, aa) in enumerate(node.named_args):
                    all_vals.named_args.append(
                        [nn, as_apply([all_idxs, self.build_idxs_vals(aa, all_idxs)])]
                    )
                    checkpoint()
                all_vals = vectorize_stochastic(all_vals)

                checkpoint()
                wanted_vals = scope.idxs_take(
                    all_idxs,  # -- may grow in future
                    all_vals,  # -- may be replaced in future
                    wanted_idxs,
                )  # -- fixed.
                if node in self.idxs_memo:
                    assert self.idxs_memo[node].name == "array_union"
                    self.idxs_memo[node].pos_args.append(wanted_idxs)
                    toposort(self.idxs_memo[node])
                    # -- this catches the cycle bug mentioned above
                    for take in self.take_memo[node]:
                        assert take.name == "idxs_take"
                        take.pos_args[1] = all_vals
                    self.take_memo[node].append(wanted_vals)
                else:
                    self.idxs_memo[node] = all_idxs
                    self.take_memo[node] = [wanted_vals]
                checkpoint()

        return wanted_vals

    def idxs_by_label(self):
        return {name: self.idxs_memo[node] for name, node in list(self.params.items())}

    def vals_by_label(self):
        return {
            name: self.take_memo[node][0].pos_args[1]
            for name, node in list(self.params.items())
        }
