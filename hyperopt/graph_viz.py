"""
Use graphviz's dot language to express the relationship between hyperparamters
in a search space.

"""
from future import standard_library

import io
from .pyll_utils import expr_to_config

standard_library.install_aliases()


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
    >>> open('foo.dot', 'w').write(dot_hyperparameters(search_space()))

    Then later from the shell, type e.g.
    dot -Tpng foo.dot > foo.png && eog foo.png

    Graphviz has other tools too: http://www.graphviz.org

    """
    conditions = ()
    hps = {}
    expr_to_config(expr, conditions, hps)
    rval = io.StringIO()
    print("digraph {", file=rval)
    edges = set()

    def var_node(a):
        print('"%s" [ shape=box];' % a, file=rval)

    def cond_node(a):
        print('"%s" [ shape=ellipse];' % a, file=rval)

    def edge(a, b):
        text = f'"{a}" -> "{b}";'
        if text not in edges:
            print(text, file=rval)
            edges.add(text)

    for hp, dct in list(hps.items()):
        # create the node
        var_node(hp)

        # create an edge from anything it depends on
        for and_conds in dct["conditions"]:
            if len(and_conds) > 1:
                parent_label = " & ".join(
                    ["%(name)s%(op)s%(val)s" % cond.__dict__ for cond in and_conds]
                )
                cond_node(parent_label)
                edge(parent_label, hp)
                for cond in and_conds:
                    sub_parent_label = f"{cond.name}{cond.op}{cond.val}"
                    cond_node(sub_parent_label)
                    edge(cond.name, sub_parent_label)
                    edge(sub_parent_label, parent_label)
            elif len(and_conds) == 1:
                parent_label = "{}{}{}".format(
                    and_conds[0].name,
                    and_conds[0].op,
                    and_conds[0].val,
                )
                edge(and_conds[0].name, parent_label)
                cond_node(parent_label)
                edge(parent_label, hp)
    print("}", file=rval)
    return rval.getvalue()
