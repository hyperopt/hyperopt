from itertools import izip

import sys

import numpy as np

from pyll import delayed
from pyll.base import array_union, repeat
from pyll import stochastic
from hyperopt.pyll.partial import depth_first_traversal, topological_sort
from hyperopt.pyll.partial import PartialPlus, as_partialplus


stoch = stochastic.implicit_stochastic_symbols


def ERR(msg):
    print >> sys.stderr, 'hyperopt.vectorize.ERR', msg


def _group_ids_by_choice(idxs, choices, n_options):
    """
    WRITEME

    Parameters
    -----------
    idxs : list
        List of the IDs available at the time of split.
    choices : list
        List of integers (same length as idxs) that indicates which
        choice was made for the corresponding element of `idxs`.
    n_options : int
        The number of possible choices at this choice node.
        `max(choices) < n_options`.

    Returns
    -------
    rval : list of lists
        List of `n_options`  different lists that maps choices to
        which IDs made those choices.
    """
    # Empty list
    rval = [[] for ii in xrange(n_options)]
    assert len(idxs) == len(choices)
    if len(idxs) != len(choices):
        raise ValueError('idxs and choices have different lengths',
                         (len(idxs), len(choices)))
    for index, choice in izip(idxs, choices):
        rval[choice].append(index)
    return rval


def _collect_values_for_idxs(idxs, choices, *vals):
    """
    WRITEME

    Parameters
    ----------
    idxs : list
        List of the IDs available at the time of split.
    choices : list
        List of integers (same length as idxs) that indicates which
        choice was made for the corresponding element of `idxs`.
    vals : tuple
        List of pairs (length = number of possible choices), where the
        first element of every pair is a list of indices (trials where
        that choice was made), and the second element is a list of
        arbitrary values associated with the corresponding IDs.

    Returns
    -------
    rval : list
        Returns a list of values from vals in the order specified by idxs.
    """
    rval = []
    assert len(idxs) == len(choices)
    for idx, choice_idx in izip(idxs, choices):
        v_ids, v_values = vals[choice_idx]
        # TODO: This implementation is rather inefficient.
        rval.append(v_values[list(v_ids).index(idx)])
    return rval


def idxs_map(idxs, cmd, *args, **kwargs):
    """
    Return the cmd applied at positions idxs, by retrieving args and kwargs
    from the (idxs, vals) pair elements of `args` and `kwargs`.

    N.B. args and kwargs may generally include information for more idx values
    than are requested by idxs.

    Parameters
    ----------
    idxs : list
        List of the IDs available at the time of split.
    args : list of tuples
        List of pairs, where each pair is `(list_of_ids, list_of_vals)`.
    kwargs : dict
        Dictionary mapping argument names to pairs, where each pair is
        `(list_of_ids, list_of_vals)`.

    Returns
    -------
    mapped_cmd : list
        List of results corresponding to applying `cmd` to each ID in
        `idxs` by selecting the appropriate arguments in `*args` and
        `**kwargs`.

    Notes
    -----
    Could be special cased if every `list_of_ids` is the same object
    as `idxs`.
    """
    # XXX: consider insisting on sorted idxs
    # XXX: use np.searchsorted instead of dct

    if 0:  # these should all be true, but evaluating them is slow
        for ii, (idxs_ii, vals_ii) in enumerate(args):
            for jj in idxs:
                assert jj in idxs_ii
        for kw, (idxs_kw, vals_kw) in kwargs.iteritems():
            for jj in idxs:
                assert jj in idxs_kw

    args_imap = []
    for idxs_j, vals_j in args:
        if len(idxs_j):
            args_imap.append(dict(izip(idxs_j, vals_j)))
        else:
            args_imap.append({})

    kwargs_imap = {}
    for kw, (idxs_j, vals_j) in kwargs.iteritems():
        if len(idxs_j):
            kwargs_imap[kw] = dict(izip(idxs_j, vals_j))
        else:
            kwargs_imap[kw] = {}

    # getattr(delayed, cmd)
    # Figure out what created "cmd" in the first place
    # if isinstance(cmd, basestring) do something, otherwise just use it?
    f = cmd
    rval = []
    for ii in idxs:
        try:
            args_nn = [arg_imap[ii] for arg_imap in args_imap]
        except Exception:
            ERR('args_nn %s' % cmd)
            ERR('ii %s' % ii)
            ERR('arg_imap %s' % str(arg_imap))
            ERR('args_imap %s' % str(args_imap))
            raise
        try:
            kwargs_nn = dict((kw, arg_imap[ii])
                             for kw, arg_imap in kwargs_imap.iteritems())
        except Exception:
            ERR('args_nn %s' % cmd)
            ERR('ii %s' % ii)
            ERR('kw %s' % kw)
            ERR('arg_imap %s' % str(arg_imap))
            raise
        try:
            rval_nn = f(*args_nn, **kwargs_nn)
        except Exception:
            ERR('error calling impl of %s' % cmd)
            raise
        rval.append(rval_nn)
    return rval


def idxs_take(idxs, vals, which):
    """
    Return `vals[which]` where `which` is a subset of `idxs`
    """
    # TODO: consider insisting on sorted idxs
    # TODO: use np.searchsorted instead of dct
    assert len(idxs) == len(vals)
    table = dict(izip(idxs, vals))
    return np.asarray([table[w] for w in which])


def uniq(lst):
    """
    Return unique elements of `lst`, preserving order.

    TODO: make this simpler.
    """
    s = set()
    rval = []
    for l in lst:
        if id(l) not in s:
            s.add(id(l))
            rval.append(l)
    return rval


def vectorize_stochastic(orig):
    """
    orig : pyll graph
        Equivalent to an `idx_map` over a stochastic node (in
        implicit_stochastic_symbols)

    Returns
    -------
    alternate_graph : pyll graph
        pyll graph that has had stochastic nodes swapped out for vectorized
        versions.
    """
    if orig.name == 'idxs_map' and orig.args[1].value in stoch:
        # -- this is an idxs_map of a random draw of distribution `dist`
        idxs = orig.args[0]
        dist = orig.args[1].value

        def foo(arg):  # TODO: ffs, better name
            # -- each argument is an idxs, vals pair
            # 'pos_args' call is not really a thing anymore
            #assert arg.name == 'pos_args'
            assert len(arg.args) == 2
            arg_vals = arg.args[1]

            # XXX: write a pattern-substitution rule for this case
            if arg_vals.name == 'idxs_take':
                if arg_vals.arg['vals'].name == 'asarray':
                    if arg_vals.arg['vals'].inputs()[0].name == 'repeat':
                        # -- draws are iid, so forget about
                        #    repeating the distribution parameters
                        tmp = arg_vals.arg['vals'].inputs()[0]
                        repeated_thing = tmp.inputs()[1]
                        return repeated_thing
            if arg.args[0] is idxs:
                return arg_vals
            else:
                # -- arg.args[0] is a superset of idxs
                #    TODO: slice out correct elements using
                #    idxs_take, but more importantly - test this case.
                raise NotImplementedError()
        new_pos_args = [foo(arg) for arg in orig.args[2:]]
        new_named_args = dict((aname, foo(arg))
                              for aname, arg in orig.keywords.iteritems())
        vnode = PartialPlus(dist, *new_pos_args, **new_named_args)
        n_times = delayed.len(idxs)
        if 'size' in vnode.keywords:
            raise NotImplementedError('random node already has size')
        vnode.keywords['size'] = n_times
        return vnode
    else:
        return orig


def replace_repeat_stochastic(expr, return_memo=False):
    memo = {}
    for ii, orig in enumerate(depth_first_traversal(expr)):
        if orig.name == 'idxs_map' and orig.args[1].value in stoch:
            # -- this is an idxs_map of a random draw of distribution `dist`
            idxs = orig.args[0]
            dist = orig.args[1].value

            def foo(arg):  # TODO: ffs, better name
                # Never getting called
                assert False
                # -- each argument is an idxs, vals pair
                assert arg.name == 'pos_args'
                assert len(arg.args) == 2
                arg_vals = arg.args[1]
                if (arg_vals.name == 'asarray'
                        and arg_vals.inputs()[0].name == 'repeat'):
                    # -- draws are iid, so forget about
                    #    repeating the distribution parameters
                    repeated_thing = arg_vals.inputs()[0].inputs()[1]
                    return repeated_thing
                else:
                    if arg.args[0] is idxs:
                        return arg_vals
                    else:
                        # -- arg.args[0] is a superset of idxs
                        #    TODO: slice out correct elements using
                        #    idxs_take, but more importantly - test this case.
                        raise NotImplementedError()
            print "***** orig.args *******"
            print orig.args

            new_pos_args = [foo(arg) for arg in orig.args[2:]]
            new_named_args = dict((aname, foo(arg))
                                  for aname, arg in orig.keywords.iteritems())
            vnode = PartialPlus(dist, *new_pos_args, **new_named_args)
            n_times = delayed.len(idxs)
            if 'size' in vnode.keywords:
                raise NotImplementedError('random node already has size')
            vnode.keywords['size'] = n_times
            # -- loop over all nodes that *use* this one, and change them
            for client in nodes[ii + 1:]:
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
    directly. It is meant to serve as the input to a suggest algo.

    node_to_symbolic_ids - node in expr graph -> all elements we might need for
    it node_to_idxs_takes - node in expr graph -> all exprs retrieving computed
    elements

    idxs memo - maps every node in the original expression graph to a symbolic
    list of ids that are needed for it

    node_to_idxs_takes - node in expression -> instantiations for different ids

    expr : original pyll graph
    expr_idxs : symbolic list of ids that we might ask for
    v_expr : vectorized version of expr, basically a map of evaluate, never
    actually run
    params : dictionary that maps name of each hyperparameter to corresponding
    node in original pyll graph

    """

    def __init__(self, expr, expr_idxs, build=True):
        self.expr = expr
        self.expr_idxs = expr_idxs
        self.dfs_nodes = list(depth_first_traversal(expr))
        self.params = {}
        for ii, node in enumerate(self.dfs_nodes):
            if node.name == 'hyperopt_param':
                label = node.arg['label'].value
                self.params[label] = node.arg['obj']
        # -- recursive construction
        #    This makes one term in each idxs, vals memo for every
        #    directed path through the switches in the graph.

        self.node_to_symbolic_ids = {}  # node -> union, all idxs computed

        # TODO: Better name
        self.node_to_idxs_takes = {}
        self.v_expr = self.build_idxs_vals(expr, expr_idxs)

        #TODO: graph-optimization pass to remove cruft:
        #  - unions of 1
        #  - unions of full sets with their subsets
        #  - idxs_take that can be merged

        self.assert_integrity_idxs_take()

    def assert_integrity_idxs_take(self):
        node_to_symbolic_ids = self.node_to_symbolic_ids
        node_to_idxs_takes = self.node_to_idxs_takes
        after = depth_first_traversal(self.expr)
        assert list(after) == self.dfs_nodes
        assert (set(node_to_symbolic_ids.keys()) ==
                set(node_to_idxs_takes.keys()))
        for node in node_to_symbolic_ids:
            idxs = node_to_symbolic_ids[node]
            assert idxs.name == 'array_union'
            vals = node_to_idxs_takes[node][0].args[1]
            for take in node_to_idxs_takes[node]:
                assert take.name == 'idxs_take'
                assert (idxs, vals) == take.args[:2]

    def build_idxs_vals(self, node, wanted_idxs):
        """
        This recursive procedure should be called on an output-node.
        """
        checkpoint_asserts = False

        def checkpoint():
            if checkpoint_asserts:
                self.assert_integrity_idxs_take()
                if node in self.node_to_symbolic_ids:
                    topological_sort(self.node_to_symbolic_ids[node])
                if node in self.node_to_idxs_takes:
                    for take in self.node_to_idxs_takes[node]:
                        topological_sort(take)

        checkpoint()

        # wanted_idxs are fixed, whereas node_to_symbolic_ids
        # is full of unions, that can grow in subsequent recursive
        # calls to build_idxs_vals with node as argument.
        assert wanted_idxs != self.node_to_symbolic_ids.get(node)

        # -- easy exit case
        if node.name == 'hyperopt_param':
            # -- ignore, not vectorizing
            return self.build_idxs_vals(node.arg['obj'], wanted_idxs)

        # -- easy exit case
        elif node.name == 'hyperopt_result':
            # -- ignore, not vectorizing
            return self.build_idxs_vals(node.arg['obj'], wanted_idxs)

        # -- literal case: always take from universal set
        elif node.name is None and node.__class__.__name__ == 'Literal':
            if node in self.node_to_symbolic_ids:
                all_idxs, all_vals = self.node_to_idxs_takes[node][0].args[:2]
                wanted_vals = delayed.idxs_take(all_idxs, all_vals,
                                                wanted_idxs)
                self.node_to_idxs_takes[node].append(wanted_vals)
                checkpoint()
            else:
                # -- initialize node_to_symbolic_ids to full set
                all_idxs = self.expr_idxs
                n_times = delayed.len(all_idxs)
                # -- put array_union into graph for consistency, though it is
                # not necessary
                all_idxs = delayed.array_union(all_idxs)
                self.node_to_symbolic_ids[node] = all_idxs
                all_vals = delayed.np.asarray(delayed.repeat(n_times, node))
                wanted_vals = delayed.idxs_take(all_idxs, all_vals,
                                                wanted_idxs)
                assert node not in self.node_to_idxs_takes
                self.node_to_idxs_takes[node] = [wanted_vals]
                checkpoint()
            return wanted_vals

        # -- switch case: complicated
        elif node.name == 'switch':
            if (node in self.node_to_symbolic_ids
                    and wanted_idxs in self.node_to_symbolic_ids[node].args):
                # -- phew, easy case
                all_idxs, all_vals = self.node_to_idxs_takes[node][0].args[:2]
                wanted_vals = delayed.idxs_take(all_idxs, all_vals,
                                                wanted_idxs)
                self.node_to_idxs_takes[node].append(wanted_vals)
                checkpoint()
            else:
                # -- we need to add some indexes
                if node in self.node_to_symbolic_ids:
                    all_idxs = self.node_to_symbolic_ids[node]
                    assert all_idxs.name == 'array_union'
                    all_idxs.append_arg(wanted_idxs)
                else:
                    all_idxs = delayed.array_union(wanted_idxs)

                choice = node.args[0]
                all_choices = self.build_idxs_vals(choice, all_idxs)

                options = node.args[1:]
                args_idxs = delayed._group_ids_by_choice(all_idxs, all_choices,
                                                         len(options))
                pos_args = []
                for opt_ii, idxs_ii in izip(options, args_idxs):
                    pos_args.append(
                        as_partialplus([idxs_ii,
                                        self.build_idxs_vals(opt_ii, idxs_ii),
                                        ]))
                all_vals = delayed._collect_values_by_idxs(all_idxs,
                                                           all_choices,
                                                           *pos_args)

                wanted_vals = delayed.idxs_take(
                    all_idxs,  # -- may grow in future
                    all_vals,  # -- may be replaced in future
                    wanted_idxs  # -- fixed.
                )
                if node in self.node_to_symbolic_ids:
                    assert (self.node_to_symbolic_ids[node].name ==
                            'array_union')
                    self.node_to_symbolic_ids[node].append_arg(wanted_idxs)
                    for take in self.node_to_idxs_takes[node]:
                        assert take.name == 'idxs_take'
                        take.args[1] = all_vals
                    self.node_to_idxs_takes[node].append(wanted_vals)
                else:
                    self.node_to_symbolic_ids[node] = all_idxs
                    self.node_to_idxs_takes[node] = [wanted_vals]
                checkpoint()

        # -- general case
        else:
            # -- this is a general node.
            #    It is generally handled with node_to_symbolic_ids,
            #    but vectorize_stochastic may immediately transform it into
            #    a more compact form.
            if (node in self.node_to_symbolic_ids
                    and wanted_idxs in self.node_to_symbolic_ids[node].args):
                # -- phew, easy case
                for take in self.node_to_idxs_takes[node]:
                    if take.args[2] == wanted_idxs:
                        return take
                raise NotImplementedError('how did this happen?')
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
                if node in self.node_to_symbolic_ids:
                    all_idxs = self.node_to_symbolic_ids[node]

                else:
                    all_idxs = delayed.array_union(wanted_idxs)
                checkpoint()

                #replacement = DontEvaluate(node.func, *node.args,
                #                           **node.keywords)
                all_vals = delayed.idxs_map(all_idxs, node.func)
                for ii, aa in enumerate(node.args):
                    all_vals.append_arg(as_partialplus([
                        all_idxs, self.build_idxs_vals(aa, all_idxs)]))
                    checkpoint()
                for ii, (nn, aa) in enumerate(node.keywords.iteritems()):
                    all_vals.keywords[nn] = as_partialplus([
                        all_idxs, self.build_idxs_vals(aa, all_idxs)])
                    checkpoint()
                all_vals = vectorize_stochastic(all_vals)

                checkpoint()
                wanted_vals = delayed.idxs_take(
                    all_idxs,     # -- may grow in future
                    all_vals,     # -- may be replaced in future
                    wanted_idxs   # -- fixed.
                )
                if node in self.node_to_symbolic_ids:
                    assert (self.node_to_symbolic_ids[node].name ==
                            'array_union')
                    self.node_to_symbolic_ids[node].append_arg(wanted_idxs)
                    topological_sort(self.node_to_symbolic_ids[node])
                    # -- this catches the cycle bug mentioned above
                    for take in self.node_to_idxs_takes[node]:
                        assert take.name == 'idxs_take'
                        take.args[1] = all_vals
                    self.node_to_idxs_takes[node].append(wanted_vals)
                else:
                    self.node_to_symbolic_ids[node] = all_idxs
                    self.node_to_idxs_takes[node] = [wanted_vals]
                checkpoint()

        return wanted_vals

    def idxs_by_label(self):
        return dict((name, self.node_to_symbolic_ids[node])
                    for name, node in self.params.iteritems())

    def vals_by_label(self):
        return dict((name, self.node_to_idxs_takes[node][0].args[1])
                    for name, node in self.params.iteritems())
