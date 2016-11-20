"""
Functions to visualize an Experiment.

"""
from __future__ import print_function
import math
import sys
from past.builtins import xrange
from hyperopt.mongoexp import MongoJobs
try:
    unicode = unicode
except NameError:
    basestring = (str, bytes)
else:
    basestring = basestring
# -- don't import this here because it locks in the backend
#    and we want the unittests to be able to set the backend
# TODO: this is really bad style, create a backend plotting
# module for this that defaults to matplotlib.
# import matplotlib.pyplot as plt

import numpy as np
from . import base
from .base import miscs_to_idxs_vals

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

default_status_colors = {
    base.STATUS_NEW: 'k',
    base.STATUS_RUNNING: 'g',
    base.STATUS_OK: 'b',
    base.STATUS_FAIL: 'r'}


def algo_as_str(algo):
    if isinstance(algo, basestring):
        return algo
    return str(algo)


def main_plot_history(trials, bandit=None, algo=None, do_show=True,
                      status_colors=None):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    # self is an Experiment
    if status_colors is None:
        status_colors = default_status_colors

    # XXX: show the un-finished or error trials
    Ys, colors = zip(*[(y, status_colors[s])
                       for y, s in zip(trials.losses(bandit), trials.statuses(bandit))
                       if y is not None])
    plt.scatter(range(len(Ys)), Ys, c=colors)
    plt.xlabel('time')
    plt.ylabel('loss')

    if bandit is not None and bandit.loss_target is not None:
        plt.axhline(bandit.loss_target)
        ymin = min(np.min(Ys), bandit.loss_target)
        ymax = max(np.max(Ys), bandit.loss_target)
        yrange = ymax - ymin
        ymean = (ymax + ymin) / 2.0
        plt.ylim(
            ymean - 0.53 * yrange,
            ymean + 0.53 * yrange,
        )
    best_err = trials.average_best_error(bandit)
    print("avg best error:", best_err)
    plt.axhline(best_err, c='g')

    plt.title('bandit: %s algo: %s' % (
        bandit.short_str() if bandit else '-',
        algo_as_str(algo)))
    if do_show:
        plt.show()


def main_plot_histogram(trials, bandit=None, algo=None, do_show=True):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    status_colors = default_status_colors
    Xs, Ys, Ss, Cs = zip(*[(x, y, s, status_colors[s])
                           for (x, y, s) in zip(trials.specs, trials.losses(bandit),
                                                trials.statuses(bandit))
                           if y is not None])

    # XXX: deal with ok vs. un-finished vs. error trials
    print('Showing Histogram of %i jobs' % len(Ys))
    plt.hist(Ys)
    plt.xlabel('loss')
    plt.ylabel('frequency')

    plt.title('bandit: %s algo: %s' % (
        bandit.short_str() if bandit else '-',
        algo_as_str(algo)))
    if do_show:
        plt.show()


def main_plot_vars(trials, bandit=None, do_show=True, fontsize=10,
                   colorize_best=None,
                   columns=5,
                   ):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    idxs, vals = miscs_to_idxs_vals(trials.miscs)
    losses = trials.losses()
    finite_losses = [y for y in losses if y not in (None, float('inf'))]
    asrt = np.argsort(finite_losses)
    if colorize_best != None:
        colorize_thresh = finite_losses[asrt[colorize_best + 1]]
    else:
        # -- set to lower than best (disabled)
        colorize_thresh = finite_losses[asrt[0]] - 1

    loss_min = min(finite_losses)
    loss_max = max(finite_losses)
    print('finite loss range', loss_min, loss_max, colorize_thresh)

    loss_by_tid = dict(zip(trials.tids, losses))

    def color_fn(lossval):
        if lossval is None:
            return (1, 1, 1)
        else:
            t = 4 * (lossval - loss_min) / (loss_max - loss_min + .0001)
            if t < 1:
                return t, 0, 0
            if t < 2:
                return 2 - t, t - 1, 0
            if t < 3:
                return 0, 3 - t, t - 2
            return 0, 0, 4 - t

    def color_fn_bw(lossval):
        if lossval in (None, float('inf')):
            return (1, 1, 1)
        else:
            t = (lossval - loss_min) / (loss_max - loss_min + .0001)
            if lossval < colorize_thresh:
                return (0., 1. - t, 0.)  # -- red best black worst
            else:
                return (t, t, t)    # -- white=worst, black=best

    all_labels = list(idxs.keys())
    titles = ['%s (%s)' % (label, bandit.params[label].name)
              for label in all_labels]
    order = np.argsort(titles)

    C = columns
    R = int(np.ceil(len(all_labels) / float(C)))

    for plotnum, varnum in enumerate(order):
        label = all_labels[varnum]
        plt.subplot(R, C, plotnum + 1)

        # hide x ticks
        ticks_num, ticks_txt = plt.xticks()
        plt.xticks(ticks_num, ['' for i in xrange(len(ticks_num))])

        dist_name = bandit.params[label].name
        x = idxs[label]
        if 'log' in dist_name:
            y = np.log(vals[label])
        else:
            y = vals[label]
        plt.title(titles[varnum], fontsize=fontsize)
        c = list(map(color_fn_bw, [loss_by_tid[ii] for ii in idxs[label]]))
        if len(y):
            plt.scatter(x, y, c=c)
        if 'log' in dist_name:
            nums, texts = plt.yticks()
            plt.yticks(nums, ['%.2e' % np.exp(t) for t in nums])

    if do_show:
        plt.show()


if 0:
    def erf(x):
        """Erf impl that doesn't require scipy.
        """
        # from http://www.math.sfu.ca/~cbm/aands/frameindex.htm
        # via
        # http://stackoverflow.com/questions/457408/
        #      is-there-an-easily-available-implementation-of-erf-for-python
        #
        #

        # save the sign of x
        sign = 1
        if x < 0:
            sign = -1
        x = abs(x)

        # constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return sign * y  # erf(-x) = -erf(x)

    def mixed_max_erf(scores, n_valid):
        scores = list(scores)  # shallow copy
        scores.sort()         # sort the copy
        scores.reverse()      # reverse the order

        # this is valid for classification
        # where the scores are the means of Bernoulli variables.
        best_mean = scores[0][0]
        best_variance = best_mean * (1.0 - best_mean) / (n_valid - 1)

        rval = 0.0
        rval_denom = 0.0

        for i, (vscore, tscore) in enumerate(scores):
            mean = vscore
            variance = mean * (1.0 - mean) / (n_valid - 1)
            diff_mean = mean - best_mean
            diff_variance = variance + best_variance
            # for scores, which should approach 1, the diff here will be negative (or zero).
            # so the probability of the current point being the best is the probability that
            # the current gaussian puts on positive values.
            assert diff_mean <= 0.0
            p_current_is_best = 0.5 - 0.5 * erf(-diff_mean / math.sqrt(diff_variance))
            rval += p_current_is_best * tscore
            rval_denom += p_current_is_best
            if p_current_is_best < 0.001:
                break
        return rval / rval_denom

    def mixed_max_sampled(scores, n_valid, n_samples=100, rng=None):
        scores = list(scores)  # shallow copy
        scores.sort()         # sort the copy
        scores.reverse()      # reverse the order

        # this is valid for classification
        # where the scores are the means of Bernoulli variables.
        best_mean = scores[0][0]
        best_variance = best_mean * (1.0 - best_mean) / (n_valid - 1)
        mu = []
        sigma = []
        tscores = []
        for i, (vscore, tscore) in enumerate(scores):
            mean = vscore
            variance = mean * (1.0 - mean) / (n_valid - 1)
            diff_mean = mean - best_mean
            diff_variance = variance + best_variance
            # for scores, which should approach 1, the diff here will be negative (or zero).
            # so the probability of the current point being the best is the probability that
            # the current gaussian puts on positive values.

            if -diff_mean / np.sqrt(diff_variance) > 3:
                # print 'breaking after', len(tscores), len(scores)
                break
            else:
                mu.append(diff_mean)
                sigma.append(np.sqrt(diff_variance))
                tscores.append(tscore)

        if rng is None:
            rng = np.random.RandomState(232342)

        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        tscores = np.asarray(tscores)

        nrml = rng.randn(n_samples, len(mu)) * sigma + mu
        winners = (nrml.T == nrml.max(axis=1))
        p_best_ = winners.sum(axis=0)
        p_best = p_best_ / p_best_.sum()

        # TODO: t_scores is undefined
        return np.dot(p_best, t_scores), p_best


if 0:
    def rexp_plot_acc(scores, n_valid, n_test, pbest_n_samples=100, rng=None):
        """
        Uses the current pyplot figure to show efficiency of random experiment.

        :type scores: a list of (validation accuracy, test accuracy)  pairs
        :param scores: results from the trials of a random experiment

        :type n_valid: integer
        :param n_valid: size of the validation set

        :type n_test: integer
        :param n_test: size of the test set

        :type mixed_max: function like mixed_max_erf or mixed_max_sampled
        :param mixed_max: the function to estimate the maximum of a validation sample

        """
        if rng is None:
            rng = np.random.RandomState(232342)
        K = 1
        scatter_x = []
        scatter_y = []
        scatter_c = []
        box_x = []
        log_K = 0
        while K < len(scores):
            n_batches_of_K = len(scores) // K
            if n_batches_of_K < 2:
                break

            def best_score(i):
                scores_i = scores[i * K:(i + 1) * K]
                rval = np.dot(
                    [tscore for (vscore, tscore) in scores_i],
                    # TODO: where is pbest_sampled defined?
                    pbest_sampled(
                        [vscore for (vscore, tscore) in scores_i],
                        n_valid,
                        n_samples=pbest_n_samples,
                        rng=rng))
                # print rval
                return rval

            if n_batches_of_K < 10:
                # use scatter plot
                for i in xrange(n_batches_of_K):
                    scatter_x.append(log_K + 1)
                    scatter_y.append(best_score(i))
                    scatter_c.append((0, 0, 0))
                box_x.append([])
            else:
                # use box plot
                box_x.append([best_score(i) for i in xrange(n_batches_of_K)])
            K *= 2
            log_K += 1
        plt.scatter(scatter_x, scatter_y, c=scatter_c, marker='+', linewidths=0.2,
                    edgecolors=scatter_c)
        boxplot_lines = plt.boxplot(box_x)
        for key in boxplot_lines:
            plt.setp(boxplot_lines[key], color='black')
        # plt.setp(boxplot_lines['medians'], color=(.5,.5,.5))

        # draw the spans
        #
        # the 'test performance of the best model' is a mixture of gaussian-distributed quantity
        # with components comp_mean, and comp_var and weights w
        #
        # w[i] is prob. of i'th model being best in validation
        w = pbest_sampled([vs for (vs, ts) in scores], n_valid, n_samples=pbest_n_samples, rng=rng)
        comp_mean = np.asarray([ts for (vs, ts) in scores])
        comp_var = (comp_mean * (1 - comp_mean)) / (n_test - 1)

        # the mean of the mixture is
        mean = np.dot(w, comp_mean)

        # the variance of the mixture is
        var = np.dot(w, comp_mean**2 + comp_var) - mean**2

        # test average is distributed according to a mixture of gaussians, so we
        # have to use the following fo
        std = math.sqrt(var)
        # plt.axhline(mean, color=(1.0,1.0,1.0), linestyle='--', linewidth=0.1)
        # plt.axhspan(mean-1.96*std, mean+1.96*std, color=(0.5,0.5,0.5))
        plt.axhline(mean - 1.96 * std, color=(0.0, 0.0, 0.0))
        plt.axhline(mean + 1.96 * std, color=(0.0, 0.0, 0.0))

        # get margin:
        if 0:
            margin = 1.0 - mean
            plt.ylim(0.5 - margin, 1.0)

        # set ticks
        ticks_num, ticks_txt = plt.xticks()
        plt.xticks(ticks_num, ['%i' % (2**i) for i in xrange(len(ticks_num))])

    def rexp_pairs_raw(x, y, vscores):
        if len(x) != len(y):
            raise ValueError()
        if len(x) != len(vscores):
            raise ValueError()

        vxy = zip(vscores, x, y)
        vxy.sort()
        vscores, x, y = zip(*vxy)

        vscores = np.asarray(vscores)

        max_score = vscores.max()
        min_score = vscores.min()
        colors = np.outer(0.9 - 0.89 * (vscores - min_score) / (max_score - min_score), [1, 1, 1])
        plt.scatter(x, y, c=colors, marker='o', linewidths=0.1)

        # remove ticks labels
        nums, texts = plt.xticks()
        plt.xticks(nums, [''] * len(nums))
        nums, texts = plt.yticks()
        plt.yticks(nums, [''] * len(nums))

    class CoordType(object):
        pass

    class RealCoord(CoordType):

        @staticmethod
        def preimage(x):
            return np.asarray(x)

    class LogCoord(CoordType):

        @staticmethod
        def preimage(x):
            return np.log(x)

    class Log0Coord(CoordType):

        @staticmethod
        def preimage(x):
            x = np.asarray(x)
            return np.log(x + (x == 0) * x.min() / 2)
    IntCoord = RealCoord
    LogIntCoord = LogCoord

    class CategoryCoord(CoordType):

        def __init__(self, categories=None):
            self.categories = categories

        def preimage(self, x):
            if self.categories:
                return np.asarray([self.categories.index(xi) for xi in x])
            else:
                return x

    def rexp_pairs(x, y, vscores, xtype, ytype):
        return rexp_pairs_raw(xtype.preimage(x), ytype.preimage(y), vscores)

    class MultiHistory(object):
        """
        Show the history of multiple optimization algorithms.
        """

        def __init__(self):
            self.histories = []

        def add_experiment(self, mj, y_fn, start=0, stop=sys.maxint,
                           color=None,
                           label=None):
            trials = [(job['book_time'], job, y_fn(job))
                      for job in mj if ('book_time' in job and
                                        y_fn(job) is not None and
                                        np.isfinite(y_fn(job)))]
            trials.sort()
            trials = trials[start:stop]
            if trials:
                self.histories.append((
                    [t[1] for t in trials],
                    [t[2] for t in trials],
                    color, label))
            else:
                print('NO TRIALS')

        def add_scatters(self):
            for t, y, c, l in self.histories:
                print('setting label', l)
                plt.scatter(
                    np.arange(len(y)),
                    y,
                    c=c,
                    label=l,
                    s=12)

        def main_show(self, title=None):
            self.add_scatters()
            if title:
                plt.title(title)
            # plt.axvline(25) # make a parameter
            # plt.axhline(.2)
            # plt.axhline(.3)
            plt.show()

    def main_plot_histories(cls):
        import plotting
        conn_str_template = sys.argv[2]
        algos = sys.argv[3].split(',')
        dataset_name = sys.argv[4]
        start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        stop = int(sys.argv[6]) if len(sys.argv) > 6 else sys.maxint
        mh = plotting.MultiHistory()
        colors = ['r', 'y', 'b', 'g', 'c', 'k']

        def custom_err_fn(trial):
            if 2 == trial['status']:
                rval = 1.0 - trial['result']['best_epoch_valid']
                if rval > dict(
                        convex=.4,
                        mnist_rotated_background_images=2)[dataset_name]:
                    return None
                else:
                    return rval

        for c, algo in zip(colors, algos):
            conn_str = conn_str_template % (algo, dataset_name)
            print('algo', algo)
            mh.add_experiment(
                mj=MongoJobs.new_from_connection_str(conn_str),
                y_fn=custom_err_fn,
                color=c,
                label=algo,
                start=start,
                stop=stop)
        plt = plotting.plt
        # TODO: icml07 undefined
        plt.axhline(
            1.0 - icml07.dbn3_scores[dataset_name],
            c='k', label='manual+grid')  # , dashes=[0,1])
        mh.add_scatters()
        plt.legend()
        plt.title(dataset_name)
        plt.show()

    class ScatterByConf(object):
        trial_color_dict = {0: 'k', 1: 'g', 2: 'b', 3: 'r'}

        def __init__(self, conf_template, confs, status, y):
            self.conf_template = conf_template
            self.confs = confs
            self.y = np.asarray(y)
            assert self.y.ndim == 1
            self.status = status

            self.colors = np.asarray(
                [self.trial_color_dict.get(s, None) for s in self.status])

            self.a_choices = np.array([[e['choice']
                                        for e in t.flatten()]
                                       for t in confs])
            self.nones = np.array([[None
                                    for e in t.flatten()]
                                   for t in confs])
            self.a_names = conf_template.flatten_names()
            self.a_vars = [not np.all(self.a_choices[:, i] == self.nones[:, i])
                           for i, name in enumerate(self.a_names)]

            assert len(self.y) == len(self.a_choices)
            assert len(self.y) == len(self.colors)

        def trial_color(self, t):
            return self.trial_color_dict.get(t['status'], None)

        def scatter_one(self, column):
            assert self.a_vars[column]

            non_missing = self.a_choices[:, column] != self.nones[:, column]
            x = self.a_choices[non_missing, column]
            y = self.y[non_missing]
            c = self.colors[non_missing]
            plt.xlabel(self.a_names[column])
            plt.scatter(x, y, c=c)

        def main_show_one(self, column):
            # show all conf effects in a grid of scatter-plots
            self.scatter_one(column)
            plt.show()

        def main_show_all(self, columns=None):
            if columns == None:
                columns = range(len(self.a_vars))

            columns = [c for c in columns if c < len(self.a_vars)]

            n_vars = np.sum(self.a_vars[c] for c in columns)
            print(n_vars)
            n_rows = 1
            n_cols = 10000
            n_vars -= 1
            while n_cols > 5 and n_cols > 3 * n_rows:  # while "is ugly"
                n_vars += 1  # leave one more space at the end...
                n_rows = int(np.sqrt(n_vars))
                while n_vars % n_rows:
                    n_rows -= 1
                n_cols = n_vars / n_rows
            print(n_rows, n_cols)

            subplot_idx = 0
            for var_idx in columns:
                if self.a_vars[var_idx]:
                    plt.subplot(n_rows, n_cols, subplot_idx + 1)
                    self.scatter_one(var_idx)
                    subplot_idx += 1
            plt.show()

    def main_plot_scatter(self, argv):
        low_col = int(argv[0])
        high_col = int(argv[1])
        # upgrade jobs in db to ht_dist2-compatible things
        scatter_by_conf = ScatterByConf(
            self.bandit.template,
            self.trials,
            status=self.statuses(),
            y=self.losses())
        return scatter_by_conf.main_show_all(range(low_col, high_col))
