"""
"""
import logging
logger = logging.getLogger(__name__)
import sys
from copy import copy
from theano.gof import graph, utils, toolbox, destroyhandler


class InconsistencyError(Exception):
    """
    This exception should be thrown by listeners to Env when the
    graph's state is invalid.
    """
    pass


class Newest(object):
    def __init__(self):
        self.replacements_d = {}

    def on_attach(self, env):
        if getattr(self, 'env', env) is not env:
            raise toolbox.AlreadyThere('already attached')
        self.env = env
        if hasattr(env, 'newest') or hasattr(env, 'replacement_path'):
            raise toolbox.AlreadyThere("Newest feature is already present"
                    " or in conflict with another plugin.")
        env.newest = self.newest
        env.replacement_path = self.replacement_path

    def on_detach(self, env):
        pass

    def on_change_input(self, env, node, i, r, new_r, reason=None):
        # could be either creating or deleting a replacement
        if self.replacements_d.get(new_r, None) == r:
            # deleting
            del self.replacements_d[new_r]
        elif self.replacements_d.get(r, new_r) is not new_r:
            raise ValueError('r has already been replaced', r)
        else:
            self.replacements_d[r] = new_r

    def replacement_path(self, v):
        """Returns the sequence of replacements of v
        """
        env = self.env
        # equiv_d maps from pre-cloned variables to env variables
        if v in env.equiv:
            v = env.equiv[v]
            rval = [v]
        else:
            rval = []
        while v in self.replacements_d:
            v = self.replacements_d[v]
            rval.append[v]
        return rval

    def newest(self, v):
        return self.replacement_path(v)[-1]


class Env(utils.object2):
    """
    An Env represents a subgraph bound by a set of input variables and a
    set of output variables. The inputs list should contain all the inputs
    on which the outputs depend. Variables of type Value or Constant are
    not counted as inputs.

    The Env supports the replace operation which allows to replace a
    variable in the subgraph by another, e.g. replace (x + x).out by (2
    * x).out. This is the basis for optimization in theano.

    It can also be "extended" using env.extend(some_object). See the
    toolbox and ext modules for common extensions.

    Features added with the`extend` function can handle the following events:

    - feature.on_attach(env)
        Called by extend. The feature has great freedom in what
        it can do with the env: it may, for example, add methods
        to it dynicamically.

    - feature.on_detach(env)
        Called by remove_feature(feature).  Should remove any dynamically-added
        functionality that it installed into the env.

    - feature.on_import(env, node)*
        Called whenever a node is imported into env, which is
        just before the node is actually connected to the graph.

    - feature.on_prune(env, node)*
        Called whenever a node is pruned (removed) from the env,
        after it is disconnected from the graph.

    - feature.on_change_input(env, node, i, r, new_r, [reason=None])*
        Called whenever node.inputs[i] is changed from r to new_r.
        At the moment the callback is done, the change has already
        taken place.

    - feature.orderings(env)
        Called by toposort. It should return a dictionary of
        {node: predecessors} where predecessors is a list of
        nodes that should be computed before the key node.

        * If you raise an exception in the functions marked with an
          asterisk, the state of the graph might be inconsistent.

    - feature.on_setup_node(env, node):
        WRITEME

    - feature.on_setup_variable(env, variable):
        WRITEME

    """

    ### Special ###
    # TODO: document which things that features can do to the env

    def __init__(self, equiv, features):
        """
        Create an Env which operates on the subgraph bound by the inputs and outputs
        sets.

        This class keeps a pointer to the inputs and outputs, and also modifies them.

        #TODO: document what variables are[not] set in the env when a feature is added via the
        constructor.  How constructed is the env?

        """
        # equiv maps from external things (e.g. pre-clone) -> env variables and env nodes
        self.equiv = equiv

        self.node_locks = {}
        self.variable_locks = {}

        self._features = []
        self.inputs = []
        self.outputs = []

        # All nodes in the subgraph defined by inputs and outputs are cached in nodes
        self.nodes = set()

        # Ditto for variables
        self.variables = set()

        for f in features:
            self.extend(f)

    def add_inputs(self, inputs):
        for ii in inputs:
            self.add_input(ii)

    def add_outputs(self, outputs):
        for ii in outputs:
            self.add_output(ii)

    def add_input(self, ii):
        """
        Add an owner-less variable to the Env.
        """
        if ii.owner is not None:
            raise ValueError(
                    "One of the provided inputs is the output of an apply"
                    " node. If that is okay, either discard that input's"
                    " owner or use graph.clone.")
        self.__setup_r__(ii)
        self.inputs.append(ii)

    def add_output(self, oo):
        self.__import_r__([oo])  # adds clients attribute
        oo.clients.append(('output', len(self.outputs)))
        self.outputs.append(oo)

    ### Setup a Variable ###

    def __setup_r__(self, r):
        # sets up r so it belongs to this env
        if hasattr(r, 'env') and r.env is not None and r.env is not self:
            raise Exception("%s is already owned by another env" % r)
        r.env = self
        r.clients = []
        self.variables.add(r)

    def __setup_node__(self, node):
        # sets up apply node so it belongs to this env
        if hasattr(node, 'env') and node.env is not self:
            raise Exception("%s is already owned by another env" % node)
        node.env = self
        node.deps = {}
        self.nodes.add(node)

    def disown(self):
        """ WRITEME
        Cleans up all of this Env's nodes and variables so they are not
        associated with this Env anymore.

        The Env should not be used anymore after disown is called.

        This may not clean everything this Env's features set in the
        nodes and variables. If there are no features, this should set
        them back to what they were originally.
        """
        for node in self.nodes:
            del node.env
            del node.deps
        for variable in self.variables:
            del variable.env
            del variable.clients
        self.nodes = set()
        self.variables = set()
        self.inputs = []
        self.outputs = []


    ### clients ###

    def clients(self, r):
        """
        Set of all the (node, i) pairs such that node.inputs[i] is r.
        Tell differently, a list of (node,i) such that each node have r as input at index i.
        """
        return r.clients

    def __add_clients__(self, r, new_clients):
        """ WRITEME
        r - variable
        new_clients - list of (node, i) pairs such that node.inputs[i] is r.

        Updates the list of clients of r with new_clients.
        """
        if set(r.clients).intersection(set(new_clients)):
            # XXX: logging
            logging.error('clients intersect')
            for ii, (n, i) in enumerate(r.clients):
                logging.error(' old client %i: %s, %i id=%i' % (ii, n, i, id(n)))
            for ii, (n, i) in enumerate(new_clients):
                logging.error(' new client %i: %s, %i id=%i' % (ii, n, i, id(n)))
            assert 0, 'client intersection indicates bug elsewhere'
        r.clients += new_clients

    def __remove_clients__(self, r, clients_to_remove, prune = True):
        """ WRITEME
        r -> variable
        clients_to_remove -> list of (op, i) pairs such that node.inputs[i] is not r anymore.

        Removes all from the clients list of r.
        """
        for entry in clients_to_remove:
            r.clients.remove(entry)
            if entry in r.clients:
                # XXX: logging
                print >> sys.stderr, 'ERROR: DUPLICATE CLIENT ENTRY...'
                print >> sys.stderr, '  ENTRY', repr(entry), type(entry[0])
                print >> sys.stderr, '  CLIENTS', repr(r.clients)
            assert entry not in r.clients # an op,i pair should be unique
        if not r.clients:
            if prune:
                self.__prune_r__([r])
                return False
            return True
        return False


    ### import ###

    def __import_r__(self, variables):
        # Imports the owners of the variables
        r_owner_done = set(self.nodes)
        for node in [r.owner for r in variables if r.owner is not None]:
            if node not in r_owner_done:
                r_owner_done.add(node)
                self.__import__(node)
        for r in variables:
            if not getattr(r, 'env', None) is self:
                self.__setup_r__(r)
                self.variables.add(r)
                self.execute_callbacks('on_import', r)
            else:
                assert r in self.variables

    def __import__(self, node, check = True):
        # We import the nodes in topological order. We only are interested
        # in new nodes, so we use all variables we know of as if they were the input set.
        # (the functions in the graph module only use the input set to
        # know where to stop going down)
        new_nodes = graph.io_toposort(self.variables, node.outputs)

        if check:
            for node in new_nodes:
                if hasattr(node, 'env') and node.env is not self:
                    raise Exception("%s is already owned by another env" % node)
                for r in node.inputs:
                    if hasattr(r, 'env') and r.env is not self:
                        raise Exception("%s is already owned by another env" % r)

        for node in new_nodes:
            assert node not in self.nodes
            self.__setup_node__(node)
            for output in node.outputs:
                self.__setup_r__(output)
            for i, input in enumerate(node.inputs):
                if input not in self.variables:
                    self.__setup_r__(input)
                self.__add_clients__(input, [(node, i)])
            assert node.env is self
            self.execute_callbacks('on_import', node)


    ### prune ###

    def __prune_r__(self, variables):
        # Prunes the owners of the variables.
        for node in set(r.owner for r in variables if r.owner is not None):
            self.__prune__(node)
        for r in variables:
            if not r.clients and r in self.variables:
                self.variables.remove(r)
                del r.env
                del r.clients

    def __prune__(self, node):
        if node not in self.nodes:
            raise Exception("%s does not belong to this Env and cannot be pruned." % node)
        assert node.env is self
        # If node's outputs have no clients, removes it from the graph
        # and recursively tries to prune its inputs. If at least one
        # of the op's outputs is an output to the graph or has a client
        # then __prune__ is a no-op.
        for output in node.outputs:
            # Cannot prune an op which is an output or used somewhere
            if self.clients(output) or output in self.outputs: #output in self.outputs or self.clients(output):
                return
        self.nodes.remove(node)
        self.variables.difference_update(node.outputs)
        self.execute_callbacks('on_prune', node)

        for i, input in enumerate(node.inputs):
            self.__remove_clients__(input, [(node, i)])

        del node.env
        del node.deps



    ### change input ###

    def change_input(self, node, i, new_r, reason=None):
        """WRITEME
        Changes node.inputs[i] to new_r.

        new_r.type == old_r.type must be True, where old_r is the
        current value of node.inputs[i] which we want to replace.

        For each feature that has a 'on_change_input' method, calls:
          feature.on_change_input(env, node, i, old_r, new_r, [reason])
        """
        # TODO: ERROR HANDLING FOR LISTENERS (should it complete the change or revert it?)
        if node == 'output':
            r = self.outputs[i]
            if not r.type == new_r.type:
                raise TypeError("The type of the replacement must be the same as the type of the original Variable.", r, new_r)
            self.outputs[i] = new_r
        else:
            if node.env is not self:
                raise Exception("Cannot operate on %s because it does not belong to this Env" % node)
            r = node.inputs[i]
            if not r.type == new_r.type:
                raise TypeError("The type of the replacement must be the same as the type of the original Variable.", r, new_r)
            node.inputs[i] = new_r

        self.__import_r__([new_r])
        self.__add_clients__(new_r, [(node, i)])
        prune = self.__remove_clients__(r, [(node, i)], False)
        # Precondition: the substitution is semantically valid
        # However it may introduce cycles to the graph,  in which case the
        # transaction will be reverted later.
        self.execute_callbacks('on_change_input', node, i, r, new_r, reason=reason)

        if prune:
            self.__prune_r__([r])


    ### replace ###

    def replace(self, r, new_r, reason=None):
        """Connect clients of `r` to use `new_r` instead.

        This is the main interface to manipulate the subgraph in Env.
        For every node that uses r as input, makes it use new_r instead.
        """
        if r.env is not self:
            raise Exception("Cannot replace %s because it does not belong to this Env" % r, str(reason))
        if not r.type == new_r.type:
            raise TypeError("The type of the replacement must be the same as the type of the original Variable.", r, new_r, r.type, new_r.type, str(reason))
        if r not in self.variables:
            # this variable isn't in the graph... don't raise an exception here, just return silently
            # because it makes it easier to implement some optimizations for multiple-output ops
            return

        for node, i in list(r.clients): # copy the client list for iteration
            assert (node == 'output' and self.outputs[i] is r) or (node.inputs[i] is r)
            self.change_input(node, i, new_r, reason=reason)

    def replace_all(self, pairs, reason=None):
        """Rewire clients of old nodes to new ones.

        pairs - sequence of (old, new) variable pairs
        reason - diagnostic object for debugging (Can be anything.)
        """
        for r, new_r in pairs:
            self.replace(r, new_r, reason=reason)


    ### features ###

    def extend(self, feature):
        """WRITEME
        Adds a feature to this env. The feature may define one
        or more of the following methods:

        """
        if feature in self._features:
            return # the feature is already present
        attach = getattr(feature, 'on_attach', None)
        if attach is not None:
            try:
                attach(self)
            except toolbox.AlreadyThere:
                return
        self._features.append(feature)

    def remove_feature(self, feature):
        """WRITEME
        Removes the feature from the graph.

        Calls feature.on_detach(env) if an on_detach method is defined.
        """
        try:
            self._features.remove(feature)
        except:
            return
        detach = getattr(feature, 'on_detach', None)
        if detach is not None:
            detach(self)


    ### callback utils ###

    def execute_callbacks(self, name, *args, **kwargs):
        """WRITEME
        Calls
          getattr(feature, name)(*args)
        for each feature which has a method called after name.
        """
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue

            #####HORRIBLE OPTIONAL ARGUMENT HACK
            try:
                fn(self, *args, **kwargs)
            except TypeError, e:
                if str(e) == "on_change_input() got an unexpected keyword argument 'reason'" and len(kwargs) == 1:
                    fn(self, *args)
                else:
                    raise


    def collect_callbacks(self, name, *args):
        """WRITEME
        Returns a dictionary d such that:
          d[feature] == getattr(feature, name)(*args)
        For each feature which has a method called after name.
        """
        d = {}
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            d[feature] = fn(*args)
        return d


    ### misc ###

    def toposort(self):
        """WRITEME
        Returns an ordering of the graph's Apply nodes such that:
          - All the nodes of the inputs of a node are before that node.
          - Satisfies the orderings provided by each feature that has
            an 'orderings' method.

        If a feature has an 'orderings' method, it will be called with
        this env as sole argument. It should return a dictionary of
        {node: predecessors} where predecessors is a list of nodes
        that should be computed before the key node.
        """
        if len(self.nodes) < 2:
            # optimization
            # when there are 0 or 1 nodes, no sorting is necessary
            # This special case happens a lot because the OpWiseCLinker produces
            # 1-element graphs.
            return list(self.nodes)
        env = self
        ords = self.orderings()
        order = graph.io_toposort(env.inputs, env.outputs, ords)
        return order

    def orderings(self):
        """
        Return dict d s.t. d[node] is a list of nodes that must be evaluated
        before node itself can be evaluated.

        This is used primarily by the destroyhandler feature to ensure that all
        clients of any destroyed inputs have already computed their outputs.
        """
        ords = {}
        for feature in self._features:
            if hasattr(feature, 'orderings'):
                for node, prereqs in feature.orderings(self).items():
                    ords.setdefault(node, []).extend(prereqs)
        # eliminate duplicate prereqs
        for (node,prereqs) in ords.items():
            ords[node] = list(set(prereqs))
        return ords

    def nclients(self, r):
        """WRITEME Same as len(self.clients(r))."""
        return len(self.clients(r))

#     def edge(self, r):
#         return r in self.inputs or r in self.orphans

#     def follow(self, r):
#         node = r.owner
#         if self.edge(r):
#             return None
#         else:
#             if node is None:
#                 raise Exception("what the fuck")
#             return node.inputs

    def check_integrity(self):
        """WRITEME
        Call this for a diagnosis if things go awry.
        """
        nodes = graph.ops(self.inputs, self.outputs)
        if self.nodes != nodes:
            missing = nodes.difference(self.nodes)
            excess = self.nodes.difference(nodes)
            raise Exception("The nodes are inappropriately cached. missing, in excess: ", missing, excess)
        for node in nodes:
            if node.env is not self:
                raise Exception("Node should belong to the env.", node)
            for i, variable in enumerate(node.inputs):
                if variable.env is not self:
                    raise Exception("Input of node should belong to the env.", variable, (node, i))
                if (node, i) not in variable.clients:
                    raise Exception("Inconsistent clients list.", (node, i), variable.clients)
        variables = set(graph.variables(self.inputs, self.outputs))
        if set(self.variables) != variables:
            missing = variables.difference(self.variables)
            excess = self.variables.difference(variables)
            raise Exception("The variables are inappropriately cached. missing, in excess: ", missing, excess)
        for variable in variables:
            if variable.owner is None and variable not in self.inputs and not isinstance(variable, graph.Value):
                raise Exception("Undeclared input.", variable)
            if variable.env is not self:
                raise Exception("Variable should belong to the env.", variable)
            for node, i in variable.clients:
                if node == 'output':
                    if self.outputs[i] is not variable:
                        raise Exception("Inconsistent clients list.", variable, self.outputs[i])
                    continue
                if node not in nodes:
                    raise Exception("Client not in env.", variable, (node, i))
                if node.inputs[i] is not variable:
                    raise Exception("Inconsistent clients list.", variable, node.inputs[i])

    def __str__(self):
        return "[%s]" % ", ".join(graph.as_string(self.inputs, self.outputs))

    def __repr__(self):
        return self.__str__()


    ### clone ###

    def clone(self):
        """WRITEME"""
        return self.clone_get_equiv()[0]

    def clone_get_equiv(self):
        """WRITEME"""
        equiv = graph.clone_get_equiv(self.inputs, self.outputs)
        self.check_integrity()
        e = self.__class__([equiv[i] for i in self.inputs],
                [equiv[o] for o in self.outputs])
        e.check_integrity()
        for feature in self._features:
            e.extend(feature)
        return e, equiv


class TheanoMixin(object):
    """
    Replicate part of the theano module
    """
    def function(self, *args, **kwargs):
        return theano.function(*args, **kwargs)

    def shared(self, *args, **kwargs):
        return theano.shared(*args, **kwargs)


class TensorMixin(object):
    """
    Replicate the variable-creation part of the tensor module
    """


class SparseMixin(object):
    pass


class RandomMixin(object):
    pass


def sort_replacements(replace_pairs):
    """
    Return a list of (oldvar, newvar) pairs in dependency order.

    returns: a list of [(old0, new0), (old1, new1), ...] pairs such that
    if A < B, then newA's does not depend on oldB.

    The purpose of this function is to support a sensible interpretation of
    givens when the various subgraphs they represent are tangled up and
    co-dependent.

    """
    # Suppose we're replacing vars v1 and v2,
    # but v2 appears in the ancestors of v1.
    # In this case we have to replace v2 first, and then v1.
    v_orig_ancestors = {}
    v_origs_set = set([v_orig for (v_orig, v_repl) in replace_pairs])
    for v_orig in v_origs_set:
        anc = graph.ancestors([v_orig],
                blockers=set(
                    [v for v in v_origs_set if v is not v_orig]))
        v_orig_ancestors[v_orig] = set(anc)
    def v_cmp(x, y):
        if x[0] in v_orig_ancestors[y[0]]:
            return -1
        if y[0] in v_orig_ancestors[x[0]]:
            return 1
        return 0
    rval = list(replace_pairs)
    rval.sort(v_cmp)
    return rval


class InteractiveEnv(Env, TheanoMixin, TensorMixin, SparseMixin, RandomMixin):
    def __init__(self, equiv, features):
        Env.__init__(self, equiv, features)

    def replace_all_sorted(self, replace_pairs, validate=True, reason=None):
        replacements = sort_replacements(replace_pairs)
        replacements = [(self.equiv.get(r, r), self.equiv.get(new_r, new_r))
                for (r, new_r) in replacements if r is not new_r]

        for (r, new_r) in replacements:
            if getattr(r, 'env', None) is not self:
                raise ValueError('Cannot replace a variable not in the env', r)
        if validate:
            return self.replace_all_validate(replacements)
        else:
            return self.replace_all(replacements)

    def prefer_replace(self, replace_pairs, reason=None):
        """Move clients as possible from r to new_r without creating cycle.
        """
        replacements = sort_replacements(replace_pairs)
        replacements = [(self.equiv.get(r, r), self.equiv.get(new_r, new_r))
                for (r, new_r) in replacements if r is not new_r]
        for r, new_r in replacements:
            new_ancestors = set(graph.ancestors([new_r]))
            for node, i in list(r.clients):
                if (node == 'output'
                        or any([(outvar in new_ancestors)
                            for outvar in node.outputs])):
                    # if a client is in the ancestors of new_r, then do not
                    # transfer it.  It would create a cycle, and in the case
                    # of shape nodes... it's not what we want either.
                    continue
                assert (node == 'output' and self.outputs[i] is r) or (node.inputs[i] is r)
                self.change_input(node, i, new_r, reason=reason)


def std_interactive_env(inputs, outputs, clone_inputs_and_orphans=True):
    features = []
    features.append(toolbox.ReplaceValidate())
    features.append(Newest())
    features.append(destroyhandler.DestroyHandler())
    features.append(toolbox.PreserveNames())
    equiv = graph.clone_get_equiv(inputs, outputs,
            copy_inputs_and_orphans=clone_inputs_and_orphans)
    rval = InteractiveEnv(equiv, features)
    rval.add_inputs([equiv[v] for v in inputs])
    if not clone_inputs_and_orphans:
        for v in inputs:
            assert equiv[v] is v
        assert rval.inputs == inputs
    rval.add_outputs([equiv[v] for v in outputs])
    for k, v in equiv.iteritems():
        assert v.env is rval
    return rval

