"""
XXX

"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "James Bergstra <pylearn-dev@googlegroups.com>"

import cPickle
import logging
import os

import base
import utils

logger = logging.getLogger(__name__)

from .base import SerialExperiment


def main_search():
    from optparse import OptionParser
    parser = OptionParser(
            usage="%prog [options] [<bandit> <bandit_algo>]")
    parser.add_option('--load',
            default='',
            dest="load",
            metavar='FILE',
            help="unpickle experiment from here on startup")
    parser.add_option('--save',
            default='experiment.pkl',
            dest="save",
            metavar='FILE',
            help="pickle experiment to here on exit")
    parser.add_option("--steps",
            dest='steps',
            default='100',
            metavar='N',
            help="exit after queuing this many jobs (default: 100)")
    parser.add_option("--workdir",
            dest="workdir",
            default=os.path.expanduser('~/.hyperopt.workdir'),
            help="create workdirs here",
            metavar="DIR")
    parser.add_option("--bandit-argfile",
            dest="bandit_argfile",
            default=None,
            help="path to file containing arguments bandit constructor \
                  file format: pickle of dictionary containing two keys,\
                  {'args' : tuple of positional arguments, \
                   'kwargs' : dictionary of keyword arguments}")
    parser.add_option("--bandit-algo-argfile",
            dest="bandit_algo_argfile",
            default=None,
            help="path to file containing arguments for bandit_algo "
                  "constructor.  File format is pickled dictionary containing "
                  "two keys: 'args', a tuple of positional arguments, and "
                  "'kwargs', a dictionary of keyword arguments. "
                  "NOTE: bandit is pre-pended as first element of arg tuple.")

    (options, args) = parser.parse_args()
    try:
        bandit_json, bandit_algo_json = args
    except:
        parser.print_help()
        return -1

    try:
        if not options.load:
            raise IOError()
        handle = open(options.load, 'rb')
        self = cPickle.load(handle)
        handle.close()
    except IOError:
        bandit = utils.get_obj(bandit_json, argfile=options.bandit_argfile)
        bandit_algo = utils.get_obj(bandit_algo_json,
                                    argfile=options.bandit_algo_argfile,
                                    args=(bandit,))
        self = SerialExperiment(bandit_algo)

    try:
        self.run(int(options.steps))
    finally:
        if options.save:
            cPickle.dump(self, open(options.save, 'wb'))
