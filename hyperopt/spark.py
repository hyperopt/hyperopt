from __future__ import print_function

import copy
import numbers
import threading
import time
import timeit

from hyperopt import base, fmin, Trials
from hyperopt.utils import coarse_utcnow, _get_logger, _get_random_id

try:
    from pyspark.sql import SparkSession
    _have_spark = True
except ImportError as e:
    _have_spark = False

logger = _get_logger('hyperopt-spark')


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

    def __init__(self, parallelism=None, timeout=None, spark_session=None):
        """
        :param parallelism: Maximum number of parallel trials to run,
                            i.e., maximum number of concurrent Spark tasks.
                            If set to None or and invalid value, this will be set to the number of
                            executors in your Spark cluster.
                            Hard cap at `MAX_CONCURRENT_JOBS_ALLOWED`.
                            Default: None (= number of Spark executors).
        :param timeout: Maximum time (in seconds) which fmin is allowed to take.
                        If this timeout is hit, then fmin will cancel running and proposed trials.
                        It will retain all completed trial runs and return the best result found
                        so far.
        :param spark_session: A SparkSession object. If None is passed, SparkTrials will attempt
                              to use an existing SparkSession or create a new one. SparkSession is
                              the entry point for various facilities provided by Spark. For more
                              information, visit the documentation for PySpark.
        """
        super(SparkTrials, self).__init__(exp_key=None, refresh=False)
        if not _have_spark:
            raise Exception("SparkTrials cannot import pyspark classes.  Make sure that PySpark "
                            "is available in your environment.  E.g., try running 'import pyspark'")
        if timeout is not None and (not isinstance(timeout, numbers.Number) or timeout <= 0 or
                                    isinstance(timeout, bool)):
            raise Exception("The timeout argument should be None or a positive value. "
                            "Given value: {timeout}".format(timeout=timeout))
        self._spark = SparkSession.builder.getOrCreate() if spark_session is None \
                      else spark_session
        self._spark_context = self._spark.sparkContext
        # The feature to support controlling jobGroupIds is in SPARK-22340
        self._spark_supports_job_cancelling = hasattr(self._spark_context.parallelize([1]),
                                                      "collectWithJobGroup")
        # maxNumConcurrentTasks() is a package private API
        max_num_concurrent_tasks = self._spark_context._jsc.sc().maxNumConcurrentTasks()
        self.parallelism = self._decide_parallelism(max_num_concurrent_tasks, parallelism)

        if not self._spark_supports_job_cancelling and timeout is not None:
            logger.warning(
                "SparkTrials was constructed with a timeout specified, but this Apache "
                "Spark version does not support job group-based cancellation. The timeout will be "
                "respected when starting new Spark jobs, but SparkTrials will not be able to "
                "cancel running Spark jobs which exceed the timeout.")

        self.timeout = timeout
        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None
        self.refresh()

    @staticmethod
    def _decide_parallelism(max_num_concurrent_tasks, parallelism):
        """
        Given the user-set value of parallelism, return the value SparkTrials will actually use.
        See the docstring for `parallelism` in the constructor for expected behavior.
        """
        if max_num_concurrent_tasks == 0:
            raise Exception("There are no available spark executors.  "
                            "Add workers to your Spark cluster to use SparkTrials.")
        if parallelism is None:
            parallelism = max_num_concurrent_tasks
        elif parallelism <= 0:
            logger.warning("User-specified parallelism was invalid value ({p}), so parallelism will"
                           " be set to max_num_concurrent_tasks ({c})."
                           .format(p=parallelism, c=max_num_concurrent_tasks))
            parallelism = max_num_concurrent_tasks
        elif parallelism > max_num_concurrent_tasks:
            logger.warning("User-specified parallelism ({p}) is greater than "
                           "max_num_concurrent_tasks ({c}), so parallelism will be set to "
                           "max_num_concurrent_tasks."
                           .format(p=parallelism, c=max_num_concurrent_tasks))
            parallelism = max_num_concurrent_tasks
        if parallelism > SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED:
            logger.warning("Parallelism ({p}) is being decreased to the hard cap of "
                           "SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED ({c})"
                           .format(p=parallelism, c=SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED))
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
        total_states = [base.JOB_STATE_DONE, base.JOB_STATE_ERROR, base.JOB_STATE_CANCEL]
        return self.count_by_state_unsynced(total_states)

    def delete_all(self):
        """
        Reset the Trials to init state
        """
        super(SparkTrials, self).delete_all()
        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None

    def trial_attachments(self, trial):
        raise NotImplementedError("SparkTrials does not support trial attachments.")

    def fmin(self, fn, space, algo, max_evals,
             max_queue_len,
             rstate,
             verbose,
             pass_expr_memo_ctrl,
             catch_eval_exceptions,
             return_argmin,
             show_progressbar,
             ):
        """
        This should not be called directly but is called via :func:`hyperopt.fmin`
        Refer to :func:`hyperopt.fmin` for docs on each argument
        """

        assert not pass_expr_memo_ctrl, "SparkTrials does not support `pass_expr_memo_ctrl`"
        assert not catch_eval_exceptions, "SparkTrials does not support `catch_eval_exceptions`"

        state = _SparkFMinState(self._spark, fn, space, self)

        # Will launch a dispatcher thread which runs each trial task as one spark job.
        state.launch_dispatcher()

        try:
            res = fmin(fn, space, algo, max_evals,
                       max_queue_len=max_queue_len,
                       trials=self,
                       allow_trials_fmin=False,  # -- prevent recursion
                       rstate=rstate,
                       pass_expr_memo_ctrl=None,  # not support
                       catch_eval_exceptions=catch_eval_exceptions,
                       verbose=verbose,
                       return_argmin=return_argmin,
                       points_to_evaluate=None,  # not support
                       show_progressbar=show_progressbar)
        except BaseException as e:
            logger.debug("fmin thread exits with an exception raised.")
            raise e
        else:
            logger.debug("fmin thread exits normally.")
            return res
        finally:
            state.wait_for_all_threads()

            logger.info("Total Trials: {t}: {s} succeeded, {f} failed, {c} cancelled.".format(
                t=self.count_total_trials(),
                s=self.count_successful_trials(),
                f=self.count_failed_trials(),
                c=self.count_cancelled_trials()
            ))


class _SparkFMinState:
    """
    Class for managing threads which run concurrent Spark jobs.

    This maintains a primary dispatcher thread, plus 1 thread per Hyperopt trial.
    Each trial's thread runs 1 Spark job with 1 task.
    """

    # definition of a long-running trial, configurable here for testing purposes
    _LONG_TRIAL_DEFINITION_SECONDS = 60

    def __init__(self,
                 spark,
                 eval_function,
                 space,
                 trials):
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
            interrupt_on_cancel = spark_context.getLocalProperty("spark.job.interruptOnCancel")
            if interrupt_on_cancel is None:
                self._job_interrupt_on_cancel = False
            else:
                self._job_interrupt_on_cancel = "true" == interrupt_on_cancel.lower()
            # In certain Spark deployments, the local property "spark.jobGroup.id" value is None,
            # so we create one to use for SparkTrials.
            if self._job_group_id is None:
                self._job_group_id = "Hyperopt_SparkTrials_" + _get_random_id()
            if self._job_desc is None:
                self._job_desc = "Trial evaluation jobs launched by hyperopt fmin"
            logger.debug("Job group id: {g}, job desc: {d}, job interrupt on cancel: {i}"
                         .format(g=self._job_group_id,
                                 d=self._job_desc,
                                 i=self._job_interrupt_on_cancel))

    def running_trial_count(self):
        return self.trials.count_by_state_unsynced(base.JOB_STATE_RUNNING)

    @staticmethod
    def _begin_trial_run(trial):
        trial['state'] = base.JOB_STATE_RUNNING
        now = coarse_utcnow()
        trial['book_time'] = now
        trial['refresh_time'] = now
        logger.debug("trial task {tid} started".format(tid=trial['tid']))

    def _finish_trial_run(self, is_success, is_cancelled, trial, data):
        """
        Call this method when a trial evaluation finishes. It will save results to the trial object
        and update task counters.
        :param is_success: whether the trial succeeded
        :param is_cancelled: whether the trial was cancelled
        :param data: If the trial succeeded, this is the return value from the trial task function.
                     Otherwise, this is the exception raised when running the trial task.
        """
        if is_cancelled:
            logger.debug("trial task {tid} cancelled, exception is {e}"
                         .format(tid=trial['tid'], e=str(data)))
            self._write_cancellation_back(trial, e=data)
        elif is_success:
            logger.debug("trial task {tid} succeeded, result is {r}"
                         .format(tid=trial['tid'], r=data))
            self._write_result_back(trial, result=data)
        else:
            logger.debug("trial task {tid} failed, exception is {e}"
                         .format(tid=trial['tid'], e=str(data)))
            self._write_exception_back(trial, e=data)

    def launch_dispatcher(self):
        def run_dispatcher():
            start_time = timeit.default_timer()
            last_time_trials_finished = start_time

            spark_task_maxFailures = int(self.spark.conf.get('spark.task.maxFailures', '4'))
            # When tasks take a long time, it can be bad to have Spark retry failed tasks.
            # This flag lets us message the user once about this issue if we find that tasks
            # are taking a long time to finish.
            can_warn_about_maxFailures = spark_task_maxFailures > 1

            while not self._fmin_done:
                new_tasks = self._poll_new_tasks()

                for trial in new_tasks:
                    self._run_trial_async(trial)

                cur_time = timeit.default_timer()
                elapsed_time = cur_time - start_time
                if len(new_tasks) > 0:
                    last_time_trials_finished = cur_time

                # In the future, timeout checking logic could be moved to `fmin`.
                # For now, timeouts are specific to SparkTrials.
                # When a timeout happens:
                #  - Set `trials._fmin_cancelled` flag to be True.
                #  - FMinIter checks this flag and exits if it is set to True.
                if self.trials.timeout is not None and elapsed_time > self.trials.timeout and\
                        not self.trials._fmin_cancelled:
                    self.trials._fmin_cancelled = True
                    self.trials._fmin_cancelled_reason = "fmin run timeout"
                    self._cancel_running_trials()
                    logger.warning("fmin cancelled because of " + self.trials._fmin_cancelled_reason)

                if can_warn_about_maxFailures and cur_time - last_time_trials_finished \
                        > _SparkFMinState._LONG_TRIAL_DEFINITION_SECONDS:
                    logger.warning(
                        "SparkTrials found that the Spark conf 'spark.task.maxFailures' is set to "
                        "{maxFailures}, which will make trials re-run automatically if they fail. "
                        "If failures can occur from bad hyperparameter settings, or if trials are "
                        "very long-running, then retries may not be a good idea. "
                        "Consider setting `spark.conf.set('spark.task.maxFailures', '1')` to "
                        "prevent retries.".format(maxFailures=spark_task_maxFailures))
                    can_warn_about_maxFailures = False

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
        return base.spec_from_misc(trial['misc'])

    @staticmethod
    def _write_result_back(trial, result):
        trial['state'] = base.JOB_STATE_DONE
        trial['result'] = result
        trial['refresh_time'] = coarse_utcnow()

    @staticmethod
    def _write_exception_back(trial, e):
        trial['state'] = base.JOB_STATE_ERROR
        trial['misc']['error'] = (str(type(e)), str(e))
        trial['refresh_time'] = coarse_utcnow()

    @staticmethod
    def _write_cancellation_back(trial, e):
        trial['state'] = base.JOB_STATE_CANCEL
        trial['misc']['error'] = (str(type(e)), str(e))
        trial['refresh_time'] = coarse_utcnow()

    def _run_trial_async(self, trial):
        def run_task_thread():
            local_eval_function, local_space = self.eval_function, self.space
            params = self._get_spec_from_trial(trial)

            def run_task_on_executor(_):
                domain = base.Domain(local_eval_function, local_space, pass_expr_memo_ctrl=None)
                result = domain.evaluate(params, ctrl=None, attach_attachments=False)
                yield result
            try:
                worker_rdd = self.spark.sparkContext.parallelize([0], 1)
                if self.trials._spark_supports_job_cancelling:
                    result = worker_rdd.mapPartitions(run_task_on_executor).collectWithJobGroup(
                        self._job_group_id, self._job_desc, self._job_interrupt_on_cancel
                    )[0]
                else:
                    result = worker_rdd.mapPartitions(run_task_on_executor).collect()[0]
            except BaseException as e:
                # I recommend to catch all exceptions here, it can make the program more robust.
                # There're several possible reasons lead to raising exception here.
                # so I use `except BaseException` here.
                #
                # If cancelled flag is set, it represent we need to cancel all running tasks,
                # Otherwise it represent the task failed.
                self._finish_trial_run(is_success=False, is_cancelled=self.trials._fmin_cancelled,
                                       trial=trial, data=e)
                logger.debug("trial {tid} task thread catches an exception and writes the "
                             "info back correctly."
                             .format(tid=trial['tid']))
            else:
                self._finish_trial_run(is_success=True, is_cancelled=self.trials._fmin_cancelled,
                                       trial=trial, data=result)
                logger.debug("trial {tid} task thread exits normally and writes results "
                             "back correctly."
                             .format(tid=trial['tid']))

        task_thread = threading.Thread(target=run_task_thread)
        task_thread.setDaemon(True)
        task_thread.start()
        self._task_threads.add(task_thread)

    def _poll_new_tasks(self):
        new_task_list = []
        for trial in copy.copy(self.trials.trials):
            if trial['state'] == base.JOB_STATE_NEW:
                # check parallelism limit
                if self.running_trial_count() >= self.trials.parallelism:
                    break
                new_task_list.append(trial)
                self._begin_trial_run(trial)
        return new_task_list

    def _cancel_running_trials(self):
        if self.trials._spark_supports_job_cancelling:
            logger.debug("Cancelling all running jobs in job group {g}"
                         .format(g=self._job_group_id))
            self.spark.sparkContext.cancelJobGroup(self._job_group_id)
            # Make a copy of trials by slicing
            for trial in self.trials.trials[:]:
                if trial['state'] in [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]:
                    trial['state'] = base.JOB_STATE_CANCEL
        else:
            logger.info("Because the current Apache PySpark version does not support "
                        "cancelling jobs by job group ID, SparkTrials will block until all of "
                        "its running Spark jobs finish.")

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
