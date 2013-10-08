"""
Use graphviz's dot language to express the relationship between hyperparamters
in a search space.

"""

import StringIO
from pyll_utils import expr_to_config


def dot_hyperparameters(expr):
    """
    Return a dot language specification of a graph which describes the
    relationship between hyperparameters. Each hyperparameter within the
    pyll expression `expr` is represented by a rectangular node, and
    each value of each choice node that creates a conditional variable
    in the search space is represented by an elliptical node.

    The direction of the arrows corresponds to the sequence of events
    in an ancestral sampling process.

    E.g.:
    >>> open('foo.dot', 'wb').write(dot_hyperparameters(search_space()))

    Then later from the shell, type e.g.
    dot -Tpng foo.dot > foo.png && eog foo.png

    Graphviz has other tools too: http://www.graphviz.org

    """
    conditions = ()
    hps = {}
    expr_to_config(expr, conditions, hps)
    rval = StringIO.StringIO()
    print >> rval, "digraph {"
    edges = set()

    def var_node(a):
        print >> rval, '"%s" [ shape=box];' % a

    def cond_node(a):
        print >> rval, '"%s" [ shape=ellipse];' % a

    def edge(a, b):
        text = '"%s" -> "%s";' % (a, b)
        if text not in edges:
            print >> rval, text
            edges.add(text)

    for hp, dct in hps.items():
        # create the node
        var_node(hp)

        # create an edge from anything it depends on
        for and_conds in dct['conditions']:
            if len(and_conds) > 1:
                parent_label = ' & '.join([
                    '%(name)s%(op)s%(val)s' % cond.__dict__
                    for cond in and_conds])
                cond_node(parent_label)
                edge(parent_label, hp)
                for cond in and_conds:
                    sub_parent_label = '%s%s%s' % (
                        cond.name, cond.op, cond.val)
                    cond_node(sub_parent_label)
                    edge(cond.name, sub_parent_label)
                    edge(sub_parent_label, parent_label)
            elif len(and_conds) == 1:
                parent_label = '%s%s%s' % (
                    and_conds[0].name, and_conds[0].op, and_conds[0].val)
                edge(and_conds[0].name, parent_label)
                cond_node(parent_label)
                edge(parent_label, hp)
    print >> rval, "}"
    return rval.getvalue()

