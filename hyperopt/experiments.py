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
            if not isinstance(result, (dict, base.SON)):
                raise TypeError('result should be dict-like', result)
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
    parser.add_option("--bandit_argfile",
            dest="bandit_argfile",
            default=None,
            help="path to file containing arguments bandit constructor \
                  file format: pickle of dictionary containing two keys,\
                    {'args' : tuple of positional arguments, \
                     'kwargs' : dictionary of keyword arguments}")
    parser.add_option("--bandit_algo_argfile",
            dest="bandit_algo_argfile",
            default=None,
            help="path to file containing arguments bandit_algo constructor \
                  file format: pickle of dictionary containing two keys,\
                    {'args' : tuple of positional arguments, \
                     'kwargs' : dictionary of keyword arguments}")

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
        if option.bandit_argfile:
            argfile = options.bandit_argfile
            bandit_argd = cPickle.load(open(argfile))
        else:
            bandit_argd = None
        bandit_args = bandit_argd.get('args', ())
        bandit_kwargs = bandit_argd.get('kwargs', {})      
        bandit = utils.json_call(bandit_json, 
                                 args=bandit_args,
                                 kwargs=bandit_kwargs)
        if option.bandit_algo_argfile:
            argfile = options.bandit_algo_argfile
            bandit_algo_argd = cPickle.load(open(argfile))
        else:
            bandit_algo_argd = None    
        bandit_algo_args = bandit_algo_argd.get('args', ())
        bandit_algo_kwargs = bandit_algo_argd.get('kwargs', {})            
        bandit_algo = utils.json_call(bandit_algo_json, 
                                      args=(bandit,) + bandit_algo_args,
                                      kwargs=bandit_algo_kwargs)
        self = SerialExperiment(bandit_algo)

    try:
        self.run(options.steps)
    finally:
        if options.save:
            cPickle.dump(self, open(options.save, 'wb'))
