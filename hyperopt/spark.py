import copy
import threading
import time
import timeit
import traceback

from hyperopt import base, fmin, Trials
from hyperopt.base import validate_timeout, validate_loss_threshold
from hyperopt.utils import coarse_utcnow, _get_logger, _get_random_id

from py4j.clientserver import ClientServer

try:
    from pyspark.sql import SparkSession
    from pyspark.util import VersionUtils
    import pyspark

    _have_spark = True
    _spark_major_minor_version = VersionUtils.majorMinorVersion(pyspark.__version__)
except ImportError as e:
    _have_spark = False
    _spark_major_minor_version = None

logger = _get_logger("hyperopt-spark")


class SparkTrials(Trials):
    """
    Implementation of hyperopt.Trials supporting
    distributed execution using Apache Spark clusters.
    This requires fmin to be run on a Spark cluster.

    Plugging SparkTrials into hyperopt.fmin() allows hyperopt
    to send model training and evaluation tasks to Spark workers,
    parallelizing hyperparameter search.
    Each trial (set of hyperparameter values) is handled within
    a single Spark task; i.e., each model will be fit and evaluated
    on a single worker machine.  Trials are run asynchronously.

    See hyperopt.Trials docs for general information about Trials.

    The fields we store in our trial docs match the base Trials class.  The fields include:
     - 'tid': trial ID
     - 'state': JOB_STATE_DONE, JOB_STATE_ERROR, etc.
     - 'result': evaluation result for completed trial run
     - 'refresh_time': timestamp for last status update
     - 'misc': includes:
        - 'error': (error type, error message)
     - 'book_time': timestamp for trial run start
    """

    asynchronous = True

    # Hard cap on the number of concurrent hyperopt tasks (Spark jobs) to run. Set at 128.
    MAX_CONCURRENT_JOBS_ALLOWED = 128

    def __init__(
        self, parallelism=None, timeout=None, loss_threshold=None, spark_session=None
    ):
        """
        :param parallelism: Maximum number of parallel trials to run,
                            i.e., maximum number of concurrent Spark tasks.
                            The actual parallelism is subject to available Spark task slots at
                            runtime.
                            If set to None (default) or a non-positive value, this will be set to
                            Spark's default parallelism or `1`.
                            We cap the value at `MAX_CONCURRENT_JOBS_ALLOWED=128`.
        :param timeout: Maximum time (in seconds) which fmin is allowed to take.
                        If this timeout is hit, then fmin will cancel running and proposed trials.
                        It will retain all completed trial runs and return the best result found
                        so far.
        :param spark_session: A SparkSession object. If None is passed, SparkTrials will attempt
                              to use an existing SparkSession or create a new one. SparkSession is
                              the entry point for various facilities provided by Spark. For more
                              information, visit the documentation for PySpark.
        """
        super().__init__(exp_key=None, refresh=False)
        if not _have_spark:
            raise Exception(
                "SparkTrials cannot import pyspark classes.  Make sure that PySpark "
                "is available in your environment.  E.g., try running 'import pyspark'"
            )
        validate_timeout(timeout)
        validate_loss_threshold(loss_threshold)
        self._spark = (
            SparkSession.builder.getOrCreate()
            if spark_session is None
            else spark_session
        )
        self._spark_context = self._spark.sparkContext
        self._spark_pinned_threads_enabled = isinstance(
            self._spark_context._gateway, ClientServer
        )
        # The feature to support controlling jobGroupIds is in SPARK-22340
        self._spark_supports_job_cancelling = (
            self._spark_pinned_threads_enabled
            or hasattr(self._spark_context.parallelize([1]), "collectWithJobGroup")
        )
        spark_default_parallelism = self._spark_context.defaultParallelism
        self.parallelism = self._decide_parallelism(
            requested_parallelism=parallelism,
            spark_default_parallelism=spark_default_parallelism,
        )

        if not self._spark_supports_job_cancelling and timeout is not None:
            logger.warning(
                "SparkTrials was constructed with a timeout specified, but this Apache "
                "Spark version does not support job group-based cancellation. The "
                "timeout will be respected when starting new Spark jobs, but "
                "SparkTrials will not be able to cancel running Spark jobs which exceed"
                " the timeout."
            )

        self.timeout = timeout
        self.loss_threshold = loss_threshold
        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None
        self.refresh()

    @staticmethod
    def _decide_parallelism(requested_parallelism, spark_default_parallelism):
        """
        Given the requested parallelism, return the max parallelism SparkTrials will actually use.
        See the docstring for `parallelism` in the constructor for expected behavior.
        """
        if requested_parallelism is None or requested_parallelism <= 0:
            parallelism = max(spark_default_parallelism, 1)
            logger.warning(
                "Because the requested parallelism was None or a non-positive value, "
                "parallelism will be set to ({d}), which is Spark's default parallelism ({s}), "
                "or 1, whichever is greater. "
                "We recommend setting parallelism explicitly to a positive value because "
                "the total of Spark task slots is subject to cluster sizing.".format(
                    d=parallelism, s=spark_default_parallelism
                )
            )
        else:
            parallelism = requested_parallelism

        if parallelism > SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED:
            logger.warning(
                "Parallelism ({p}) is capped at SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED ({c}).".format(
                    p=parallelism, c=SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED
                )
            )
            parallelism = SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED
        return parallelism

    def count_successful_trials(self):
        """
        Returns the current number of trials which ran successfully
        """
        return self.count_by_state_unsynced(base.JOB_STATE_DONE)

    def count_failed_trials(self):
        """
        Returns the current number of trial runs which failed
        """
        return self.count_by_state_unsynced(base.JOB_STATE_ERROR)

    def count_cancelled_trials(self):
        """
        Returns the current number of cancelled trial runs.
        This covers trials which are cancelled from exceeding the timeout.
        """
        return self.count_by_state_unsynced(base.JOB_STATE_CANCEL)

    def count_total_trials(self):
        """
        Returns the current number of all successful, failed, and cancelled trial runs
        """
        total_states = [
            base.JOB_STATE_DONE,
            base.JOB_STATE_ERROR,
            base.JOB_STATE_CANCEL,
        ]
        return self.count_by_state_unsynced(total_states)

    def delete_all(self):
        """
        Reset the Trials to init state
        """
        super().delete_all()
        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None

    def trial_attachments(self, trial):
        raise NotImplementedError("SparkTrials does not support trial attachments.")

    def fmin(
        self,
        fn,
        space,
        algo,
        max_evals,
        timeout,
        loss_threshold,
        max_queue_len,
        rstate,
        verbose,
        pass_expr_memo_ctrl,
        catch_eval_exceptions,
        return_argmin,
        show_progressbar,
        early_stop_fn,
        trials_save_file="",
    ):
        """
        This should not be called directly but is called via :func:`hyperopt.fmin`
        Refer to :func:`hyperopt.fmin` for docs on each argument
        """

        if timeout is not None:
            if self.timeout is not None:
                logger.warning(
                    "Timeout param was defined in Trials object, ignoring fmin definition"
                )
            else:
                validate_timeout(timeout)
                self.timeout = timeout

        if loss_threshold is not None:
            validate_loss_threshold(loss_threshold)
            self.loss_threshold = loss_threshold

        assert (
            not pass_expr_memo_ctrl
        ), "SparkTrials does not support `pass_expr_memo_ctrl`"
        assert (
            not catch_eval_exceptions
        ), "SparkTrials does not support `catch_eval_exceptions`"

        state = _SparkFMinState(self._spark, fn, space, self)

        # Will launch a dispatcher thread which runs each trial task as one spark job.
        state.launch_dispatcher()

        try:
            res = fmin(
                fn,
                space,
                algo,
                max_evals,
                timeout=timeout,
                loss_threshold=loss_threshold,
                max_queue_len=max_queue_len,
                trials=self,
                allow_trials_fmin=False,  # -- prevent recursion
                rstate=rstate,
                pass_expr_memo_ctrl=None,  # not supported
                catch_eval_exceptions=catch_eval_exceptions,
                verbose=verbose,
                return_argmin=return_argmin,
                points_to_evaluate=None,  # not supported
                show_progressbar=show_progressbar,
                early_stop_fn=early_stop_fn,
                trials_save_file="",  # not supported
            )
        except BaseException as e:
            logger.debug("fmin thread exits with an exception raised.")
            raise e
        else:
            logger.debug("fmin thread exits normally.")
            return res
        finally:
            state.wait_for_all_threads()

            logger.info(
                "Total Trials: {t}: {s} succeeded, {f} failed, {c} cancelled.".format(
                    t=self.count_total_trials(),
                    s=self.count_successful_trials(),
                    f=self.count_failed_trials(),
                    c=self.count_cancelled_trials(),
                )
            )


class _SparkFMinState:
    """
    Class for managing threads which run concurrent Spark jobs.

    This maintains a primary dispatcher thread, plus 1 thread per Hyperopt trial.
    Each trial's thread runs 1 Spark job with 1 task.
    """

    def __init__(self, spark, eval_function, space, trials):
        self.spark = spark
        self.eval_function = eval_function
        self.space = space
        self.trials = trials
        self._fmin_done = False
        self._dispatcher_thread = None
        self._task_threads = set()

        if self.trials._spark_supports_job_cancelling:
            spark_context = spark.sparkContext
            self._job_group_id = spark_context.getLocalProperty("spark.jobGroup.id")
            self._job_desc = spark_context.getLocalProperty("spark.job.description")
            interrupt_on_cancel = spark_context.getLocalProperty(
                "spark.job.interruptOnCancel"
            )
            if interrupt_on_cancel is None:
                self._job_interrupt_on_cancel = False
            else:
                self._job_interrupt_on_cancel = "true" == interrupt_on_cancel.lower()
            # In certain Spark deployments, the local property "spark.jobGroup.id"
            # value is None, so we create one to use for SparkTrials.
            if self._job_group_id is None:
                self._job_group_id = "Hyperopt_SparkTrials_" + _get_random_id()
            if self._job_desc is None:
                self._job_desc = "Trial evaluation jobs launched by hyperopt fmin"
            logger.debug(
                "Job group id: {g}, job desc: {d}, job interrupt on cancel: {i}".format(
                    g=self._job_group_id,
                    d=self._job_desc,
                    i=self._job_interrupt_on_cancel,
                )
            )

    def running_trial_count(self):
        return self.trials.count_by_state_unsynced(base.JOB_STATE_RUNNING)

    @staticmethod
    def _begin_trial_run(trial):
        trial["state"] = base.JOB_STATE_RUNNING
        now = coarse_utcnow()
        trial["book_time"] = now
        trial["refresh_time"] = now
        logger.debug("trial task {tid} started".format(tid=trial["tid"]))

    @staticmethod
    def _get_traceback(err):
        return err.__dict__.get("_tb_str")

    def _finish_trial_run(self, is_success, is_cancelled, trial, data):
        """
        Call this method when a trial evaluation finishes. It will save results to the
        trial object and update task counters.
        :param is_success: whether the trial succeeded
        :param is_cancelled: whether the trial was cancelled
        :param data: If the trial succeeded, this is the return value from the trial
        task function. Otherwise, this is the exception raised when running the trial
        task.
        """
        if is_cancelled:
            logger.debug(
                "trial task {tid} cancelled, exception is {e}".format(
                    tid=trial["tid"], e=str(data)
                )
            )
            self._write_cancellation_back(trial, e=data)
        elif is_success:
            logger.debug(
                "trial task {tid} succeeded, result is {r}".format(
                    tid=trial["tid"], r=data
                )
            )
            self._write_result_back(trial, result=data)
        else:
            logger.error(
                "trial task {tid} failed, exception is {e}.\n {tb}".format(
                    tid=trial["tid"], e=str(data), tb=self._get_traceback(data)
                )
            )
            self._write_exception_back(trial, e=data)

    def launch_dispatcher(self):
        def run_dispatcher():
            start_time = timeit.default_timer()

            while not self._fmin_done:
                new_tasks = self._poll_new_tasks()

                for trial in new_tasks:
                    self._run_trial_async(trial)

                cur_time = timeit.default_timer()
                elapsed_time = cur_time - start_time

                # In the future, timeout checking logic could be moved to `fmin`.
                # For now, timeouts are specific to SparkTrials.
                # When a timeout happens:
                #  - Set `trials._fmin_cancelled` flag to be True.
                #  - FMinIter checks this flag and exits if it is set to True.
                if (
                    self.trials.timeout is not None
                    and elapsed_time > self.trials.timeout
                    and not self.trials._fmin_cancelled
                ):
                    self.trials._fmin_cancelled = True
                    self.trials._fmin_cancelled_reason = "fmin run timeout"
                    self._cancel_running_trials()
                    logger.warning(
                        "fmin cancelled because of "
                        + self.trials._fmin_cancelled_reason
                    )

                time.sleep(1)

            if self.trials._fmin_cancelled:
                # Because cancelling fmin triggered, warn that the dispatcher won't launch
                # more trial tasks.
                logger.warning("fmin is cancelled, so new trials will not be launched.")

            logger.debug("dispatcher thread exits normally.")

        self._dispatcher_thread = threading.Thread(target=run_dispatcher)
        self._dispatcher_thread.setDaemon(True)
        self._dispatcher_thread.start()

    @staticmethod
    def _get_spec_from_trial(trial):
        return base.spec_from_misc(trial["misc"])

    @staticmethod
    def _write_result_back(trial, result):
        trial["state"] = base.JOB_STATE_DONE
        trial["result"] = result
        trial["refresh_time"] = coarse_utcnow()

    def _write_exception_back(self, trial, e):
        trial["state"] = base.JOB_STATE_ERROR
        trial["misc"]["error"] = (str(type(e)), self._get_traceback(e))
        trial["refresh_time"] = coarse_utcnow()

    @staticmethod
    def _write_cancellation_back(trial, e):
        trial["state"] = base.JOB_STATE_CANCEL
        trial["misc"]["error"] = (str(type(e)), str(e))
        trial["refresh_time"] = coarse_utcnow()

    def _run_trial_async(self, trial):
        def finish_trial_run(result_or_e):
            if not isinstance(result_or_e, BaseException):
                self._finish_trial_run(
                    is_success=True,
                    is_cancelled=self.trials._fmin_cancelled,
                    trial=trial,
                    data=result_or_e,
                )
                logger.debug(
                    "trial {tid} task thread exits normally and writes results "
                    "back correctly.".format(tid=trial["tid"])
                )
            else:
                self._finish_trial_run(
                    is_success=False,
                    is_cancelled=self.trials._fmin_cancelled,
                    trial=trial,
                    data=result_or_e,
                )
                logger.debug(
                    "trial {tid} task thread catches an exception and writes the "
                    "info back correctly.".format(tid=trial["tid"])
                )

        def run_task_thread():
            local_eval_function, local_space = self.eval_function, self.space
            params = self._get_spec_from_trial(trial)

            def run_task_on_executor(_):
                domain = base.Domain(
                    local_eval_function, local_space, pass_expr_memo_ctrl=None
                )

                try:
                    result = domain.evaluate(
                        params, ctrl=None, attach_attachments=False
                    )
                    yield result
                except BaseException as e:
                    # Because the traceback is not pickable, we need format it and pass it back
                    # to driver
                    _traceback_string = traceback.format_exc()
                    logger.error(_traceback_string)
                    e._tb_str = _traceback_string
                    yield e

            try:
                worker_rdd = self.spark.sparkContext.parallelize([0], 1)
                if self.trials._spark_supports_job_cancelling:
                    if self.trials._spark_pinned_threads_enabled:
                        spark_context = self.spark.sparkContext
                        spark_context.setLocalProperty(
                            "spark.jobGroup.id", self._job_group_id
                        )
                        spark_context.setLocalProperty(
                            "spark.job.description", self._job_desc
                        )
                        spark_context.setLocalProperty(
                            "spark.job.interruptOnCancel",
                            str(self._job_interrupt_on_cancel).lower(),
                        )
                        result_or_e = worker_rdd.mapPartitions(
                            run_task_on_executor
                        ).collect()[0]
                    else:
                        result_or_e = worker_rdd.mapPartitions(
                            run_task_on_executor
                        ).collectWithJobGroup(
                            self._job_group_id,
                            self._job_desc,
                            self._job_interrupt_on_cancel,
                        )[
                            0
                        ]
                else:
                    result_or_e = worker_rdd.mapPartitions(
                        run_task_on_executor
                    ).collect()[0]
            except BaseException as e:
                # I recommend to catch all exceptions here, it can make the program more robust.
                # There're several possible reasons lead to raising exception here.
                # so I use `except BaseException` here.
                #
                # If cancelled flag is set, it represent we need to cancel all running tasks,
                # Otherwise it represent the task failed.
                finish_trial_run(e)
            else:
                # The exceptions captured in run_task_on_executor would be returned in the result_or_e
                finish_trial_run(result_or_e)

        if self.trials._spark_pinned_threads_enabled:
            try:
                # pylint: disable=no-name-in-module,import-outside-toplevel
                from pyspark import inheritable_thread_target

                run_task_thread = inheritable_thread_target(run_task_thread)
            except ImportError:
                pass

        task_thread = threading.Thread(target=run_task_thread)
        task_thread.setDaemon(True)
        task_thread.start()
        self._task_threads.add(task_thread)

    def _poll_new_tasks(self):
        new_task_list = []
        for trial in copy.copy(self.trials.trials):
            if trial["state"] == base.JOB_STATE_NEW:
                # check parallelism limit
                if self.running_trial_count() >= self.trials.parallelism:
                    break
                new_task_list.append(trial)
                self._begin_trial_run(trial)
        return new_task_list

    def _cancel_running_trials(self):
        if self.trials._spark_supports_job_cancelling:
            logger.debug(
                "Cancelling all running jobs in job group {g}".format(
                    g=self._job_group_id
                )
            )
            self.spark.sparkContext.cancelJobGroup(self._job_group_id)
            # Make a copy of trials by slicing
            for trial in self.trials.trials[:]:
                if trial["state"] in [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]:
                    trial["state"] = base.JOB_STATE_CANCEL
        else:
            logger.info(
                "Because the current Apache PySpark version does not support "
                "cancelling jobs by job group ID, SparkTrials will block until all of "
                "its running Spark jobs finish."
            )

    def wait_for_all_threads(self):
        """
        Wait for the dispatcher and worker threads to finish.
        :param cancel_running_trials: If true, try to cancel all running trials.
        """
        self._fmin_done = True
        self._dispatcher_thread.join()
        self._dispatcher_thread = None
        for task_thread in self._task_threads:
            task_thread.join()
        self._task_threads.clear()
