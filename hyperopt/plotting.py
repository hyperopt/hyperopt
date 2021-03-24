"""
Functions to visualize an Experiment.

"""

import pickle

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
    base.STATUS_NEW: "k",
    base.STATUS_RUNNING: "g",
    base.STATUS_OK: "b",
    base.STATUS_FAIL: "r",
}


def main_plot_history(trials, do_show=True, status_colors=None, title="Loss History"):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    # self is an Experiment
    if status_colors is None:
        status_colors = default_status_colors

    # XXX: show the un-finished or error trials
    Ys, colors = zip(
        *[
            (y, status_colors[s])
            for y, s in zip(trials.losses(), trials.statuses())
            if y is not None
        ]
    )
    plt.scatter(range(len(Ys)), Ys, c=colors)
    plt.xlabel("time")
    plt.ylabel("loss")

    best_err = trials.average_best_error()
    print("avg best error:", best_err)
    plt.axhline(best_err, c="g")

    plt.title(title)
    if do_show:
        plt.show()


def main_plot_histogram(trials, do_show=True, title="Loss Histogram"):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    status_colors = default_status_colors
    Xs, Ys, Ss, Cs = zip(
        *[
            (x, y, s, status_colors[s])
            for (x, y, s) in zip(trials.specs, trials.losses(), trials.statuses())
            if y is not None
        ]
    )

    # XXX: deal with ok vs. un-finished vs. error trials
    print("Showing Histogram of %i jobs" % len(Ys))
    plt.hist(Ys)
    plt.xlabel("loss")
    plt.ylabel("frequency")

    plt.title(title)
    if do_show:
        plt.show()


def main_plot_vars(
    trials,
    do_show=True,
    fontsize=10,
    colorize_best=None,
    columns=5,
    arrange_by_loss=False,
):
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    idxs, vals = miscs_to_idxs_vals(trials.miscs)
    losses = trials.losses()
    finite_losses = [y for y in losses if y not in (None, float("inf"))]
    asrt = np.argsort(finite_losses)
    if colorize_best is not None:
        colorize_thresh = finite_losses[asrt[colorize_best + 1]]
    else:
        # -- set to lower than best (disabled)
        colorize_thresh = finite_losses[asrt[0]] - 1

    loss_min = min(finite_losses)
    loss_max = max(finite_losses)
    print("finite loss range", loss_min, loss_max, colorize_thresh)

    loss_by_tid = dict(zip(trials.tids, losses))

    def color_fn(lossval):
        if lossval is None:
            return 1, 1, 1
        else:
            t = 4 * (lossval - loss_min) / (loss_max - loss_min + 0.0001)
            if t < 1:
                return t, 0, 0
            if t < 2:
                return 2 - t, t - 1, 0
            if t < 3:
                return 0, 3 - t, t - 2
            return 0, 0, 4 - t

    def color_fn_bw(lossval):
        if lossval in (None, float("inf")):
            return 1, 1, 1
        else:
            t = (lossval - loss_min) / (loss_max - loss_min + 0.0001)
            if lossval < colorize_thresh:
                return 0.0, 1.0 - t, 0.0  # -- red best black worst
            else:
                return t, t, t  # -- white=worst, black=best

    all_labels = list(idxs.keys())
    titles = all_labels
    order = np.argsort(titles)

    C = min(columns, len(all_labels))
    R = int(np.ceil(len(all_labels) / float(C)))

    for plotnum, varnum in enumerate(order):
        label = all_labels[varnum]
        plt.subplot(R, C, plotnum + 1)

        # hide x ticks
        ticks_num, ticks_txt = plt.xticks()
        plt.xticks(ticks_num, [""] * len(ticks_num))

        dist_name = label

        if arrange_by_loss:
            x = [loss_by_tid[ii] for ii in idxs[label]]
        else:
            x = idxs[label]
        if "log" in dist_name:
            y = np.log(vals[label])
        else:
            y = vals[label]
        plt.title(titles[varnum], fontsize=fontsize)
        c = list(map(color_fn_bw, [loss_by_tid[ii] for ii in idxs[label]]))
        if len(y):
            plt.scatter(x, y, c=c)
        if "log" in dist_name:
            nums, texts = plt.yticks()
            plt.yticks(nums, ["%.2e" % np.exp(t) for t in nums])

    if do_show:
        plt.show()


def main_plot_1D_attachment(
    trials,
    attachment_name,
    do_show=True,
    colorize_by_loss=True,
    max_darkness=0.5,
    num_trails=None,
    preprocessing_fn=lambda x: x,
    line_width=0.1,
):
    """
    Plots trail attachments, which are 1D-Data.

    A legend is only added if the number of plotted elements is < 10.

    :param trials: The trials object to gather the attachments from.
    :param attachment_name: Thename of the attachment to gather.
    :param do_show: If the plot should be shown after creating it.
    :param colorize_by_loss: If the lines represening the trial data should be shaded by loss.
    :param max_darkness: The maximumg shading darkness (between 0 and 1). Implies colorize_by_loss=True
    :param num_trails: The number of trials to plot the attachment for. If none, all trials with a corresponding
    attachment are taken. If set to any integer value, the trials are sorted by loss and trials are selected in regular
    intervals for plotting. This ensures, that all possible outcomes are equally represented.
    :param preprocessing_fn: A preprocessing function to be appleid to the attachment before plotting.
    :param line_width: The width of the lines to be plotted.
    :return: None
    """
    # -- import here because file-level import is too early
    import matplotlib.pyplot as plt

    plt.title(attachment_name)

    lst = [l for l in trials.losses() if l is not None]
    min_loss = min(lst)
    max_loss = max(lst)

    if num_trails is None:
        plotted_trials = trials
    else:
        trials_by_loss = sorted(
            filter(lambda t: "loss" in t["result"], trials),
            key=lambda t: t["result"]["loss"],
        )
        plotted_trials = [
            trials_by_loss[i]
            for i in np.linspace(
                0, len(trials_by_loss), num_trails, endpoint=False, dtype=int
            )
        ]

    for trial in plotted_trials:
        t_attachments = trials.trial_attachments(trial)
        if attachment_name in t_attachments:
            attachment_data = np.squeeze(
                np.asanyarray(pickle.loads(t_attachments[attachment_name]))
            )
            if len(attachment_data.shape) == 1:
                attachment_data = preprocessing_fn(attachment_data)
                if colorize_by_loss:
                    color = (
                        0.0,
                        0.0,
                        0.0,
                        max_darkness
                        * (trial["result"]["loss"] - min_loss)
                        / (max_loss - min_loss),
                    )
                else:
                    color = None
                plt.plot(
                    attachment_data,
                    color=color,
                    linewidth=line_width,
                    label="loss: {:.5}".format(trial["result"]["loss"]),
                )
            else:
                pass  # TODO: warn about the skipping

    if do_show:
        if len(plotted_trials) < 10:
            plt.legend()
        plt.show()
