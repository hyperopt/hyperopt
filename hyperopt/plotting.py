"""
Functions to visualize an Experiment.

"""
from __future__ import print_function
from past.builtins import xrange
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


def main_plot_history(trials, do_show=True, status_colors=None, title="Loss History"):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    # self is an Experiment
    if status_colors is None:
        status_colors = default_status_colors

    # XXX: show the un-finished or error trials
    Ys, colors = zip(*[(y, status_colors[s])
                       for y, s in zip(trials.losses(), trials.statuses())
                       if y is not None])
    plt.scatter(range(len(Ys)), Ys, c=colors)
    plt.xlabel('time')
    plt.ylabel('loss')

    best_err = trials.average_best_error()
    print("avg best error:", best_err)
    plt.axhline(best_err, c='g')

    plt.title(title)
    if do_show:
        plt.show()


def main_plot_histogram(trials, do_show=True, title="Loss Histogram"):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    status_colors = default_status_colors
    Xs, Ys, Ss, Cs = zip(*[(x, y, s, status_colors[s])
                           for (x, y, s) in zip(trials.specs, trials.losses(),
                                                trials.statuses())
                           if y is not None])

    # XXX: deal with ok vs. un-finished vs. error trials
    print('Showing Histogram of %i jobs' % len(Ys))
    plt.hist(Ys)
    plt.xlabel('loss')
    plt.ylabel('frequency')

    plt.title(title)
    if do_show:
        plt.show()


def main_plot_vars(trials, do_show=True, fontsize=10,
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
    titles = all_labels
    order = np.argsort(titles)

    C = columns
    R = int(np.ceil(len(all_labels) / float(C)))

    for plotnum, varnum in enumerate(order):
        label = all_labels[varnum]
        plt.subplot(R, C, plotnum + 1)

        # hide x ticks
        ticks_num, ticks_txt = plt.xticks()
        plt.xticks(ticks_num, ['' for i in xrange(len(ticks_num))])

        dist_name = label
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