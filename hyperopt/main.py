#!/usr/bin/env python

"""
Entry point for bin/* scripts
"""
import sys
import logging
logger = logging.getLogger(__name__)

import datasets.main
import mongoexp

def main(cmd, fn_pos = 1):
    """
    Entry point for bin/* scripts
    XXX
    """
    logging.basicConfig(
            stream=sys.stderr,
            level=logging.INFO)
    try:
        runner = dict(
                search='main_search',
                dryrun='main_dryrun',
                plot_history='main_plot_history',
                )[cmd]
    except KeyError:
        logger.error("Command not recognized: %s" % cmd)
        # XXX: Usage message
        sys.exit(1)
    try:
        argv1 = sys.argv[fn_pos]
    except IndexError:
        logger.error('Module name required (XXX: print Usage)')
        return 1

    fn = datasets.main.load_tokens(sys.argv[fn_pos].split('.') + [runner])
    sys.exit(fn(sys.argv[fn_pos+1:]))

if __name__ == '__main__':
    cmd = sys.argv[1]
    sys.exit(main(cmd, 2))
