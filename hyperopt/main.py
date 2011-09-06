"""
Entry point for bin/* scripts
"""
import sys
import logging
logger = logging.getLogger(__name__)

import datasets.main


def main(cmd):
    """
    Entry point for bin/* scripts
    XXX
    """
    logging.basicConfig(
            stream=sys.stderr,
            level=logging.INFO)
    try:
        runner = dict(
                dryrun='main_dryrun',
                )[cmd]
    except KeyError:
        logger.error("Command not recognized: %s" % cmd)
        # XXX: Usage message
        sys.exit(1)
    try:
        argv1 = sys.argv[1]
    except IndexError:
        logger.error('Module name required (XXX: print Usage)')
        return 1

    fn = datasets.main.load_tokens(sys.argv[1].split('.') + [runner])
    sys.exit(fn())

