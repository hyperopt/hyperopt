import logging

logger = logging.getLogger(__name__)


def no_progress_loss(iteration_stop_count=20, percent_increase=0.0):
    """
    Stop function that will stop after X iteration if the loss doesn't increase

    Parameters
    ----------
    iteration_stop_count: int
        search will stop if the loss doesn't improve after this number of iteration
    percent_increase: float
        allow this percentage of variation within iteration_stop_count.
        Early stop will be triggered if the data didn't change for more than this number
        after iteration_stop_count rounds
    """

    def stop_fn(trials, best_loss=None, iteration_no_progress=0):
        new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
        if best_loss is None:
            return False, [new_loss, iteration_no_progress + 1]
        best_loss_threshold = best_loss - abs(best_loss * (percent_increase / 100.0))
        if new_loss < best_loss_threshold:
            best_loss = new_loss
            iteration_no_progress = 0
        else:
            iteration_no_progress += 1
            logger.debug(
                "No progress made: %d iteration on %d. best_loss=%.2f, best_loss_threshold=%.2f, new_loss=%.2f"
                % (
                    iteration_no_progress,
                    iteration_stop_count,
                    best_loss,
                    best_loss_threshold,
                    new_loss,
                )
            )

        return (
            iteration_no_progress >= iteration_stop_count,
            [best_loss, iteration_no_progress],
        )

    return stop_fn
