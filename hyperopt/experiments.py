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

class SerialExperiment(base.Experiment):
    """
    """

    def run(self, N):
        algo = self.bandit_algo
        bandit = algo.bandit

        for n in xrange(N):
            trial = algo.suggest(self.trials, self.results, 1)[0]
            result = bandit.evaluate(trial, base.Ctrl())
            logger.debug('trial: %s' % str(trial))
            logger.debug('result: %s' % str(result))
            self.trials.append(trial)
            self.results.append(result)


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
            default=100,
            metavar='N',
            help="exit after queuing this many jobs (default: 100)")
    parser.add_option("--workdir",
            dest="workdir",
            default=os.path.expanduser('~/.hyperopt.workdir'),
            help="create workdirs here",
            metavar="DIR")

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
        bandit = utils.json_call(bandit_json)
        bandit_algo = utils.json_call(bandit_algo_json, args=(bandit,))
        self = SerialExperiment(bandit_algo)

    try:
        self.run(options.steps)
    finally:
        if options.save:
            cPickle.dump(self, open(options.save, 'wb'))
