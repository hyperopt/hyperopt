from __future__ import print_function
from __future__ import absolute_import
import six.moves.cPickle as pickle
import os
import signal
import subprocess
import sys
import threading
import time
import unittest

import numpy as np
import nose
import nose.plugins.skip

from hyperopt.base import JOB_STATE_DONE
from hyperopt.mongoexp import parse_url
from hyperopt.mongoexp import MongoTrials
from hyperopt.mongoexp import MongoWorker
from hyperopt.mongoexp import ReserveTimeout
from hyperopt.mongoexp import as_mongo_str
from hyperopt.mongoexp import main_worker_helper
from hyperopt.mongoexp import MongoJobs
from hyperopt.fmin import fmin
from hyperopt import rand
import hyperopt.tests.test_base
from .test_domains import gauss_wave2
from six.moves import map
from six.moves import range
from six.moves import zip


def skiptest(f):
    def wrapper(*args, **kwargs):
        raise nose.plugins.skip.SkipTest()
    wrapper.__name__ = f.__name__
    return wrapper


class TempMongo(object):
    """
    Context manager for tests requiring a live database.

    with TempMongo() as foo:
        mj = foo.mongo_jobs('test1')
    """

    def __init__(self, workdir="/tmp/hyperopt_test"):
        self.workdir = workdir

    def __enter__(self):
        try:
            open(self.workdir)
            assert 0
        except IOError:
            subprocess.call(["mkdir", "-p", '%s/db' % self.workdir])
            proc_args = ["mongod",
                         "--dbpath=%s/db" % self.workdir,
                         "--noprealloc",
                         "--port=22334"]
            print("starting mongod", proc_args)
            self.mongo_proc = subprocess.Popen(
                proc_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workdir,  # this prevented mongod assertion fail
            )
            try:
                interval = .125
                while interval <= 2:
                    if interval > .125:
                        print("Waiting for mongo to come up")
                    time.sleep(interval)
                    interval *= 2
                    if self.db_up():
                        break
                if self.db_up():
                    return self
                else:
                    try:
                        os.kill(self.mongo_proc.pid, signal.SIGTERM)
                    except OSError:
                        pass  # if it crashed there is no such process
                    out, err = self.mongo_proc.communicate()
                    print(out, file=sys.stderr)
                    print(err, file=sys.stderr)
                    raise RuntimeError('No database connection', proc_args)
            except Exception as e:
                try:
                    os.kill(self.mongo_proc.pid, signal.SIGTERM)
                except OSError:
                    pass  # if it crashed there is no such process
                raise e

    def __exit__(self, *args):
        os.kill(self.mongo_proc.pid, signal.SIGTERM)
        self.mongo_proc.wait()
        subprocess.call(["rm", "-Rf", self.workdir])

    @staticmethod
    def connection_string(dbname):
        return as_mongo_str('localhost:22334/{}/jobs'.format(dbname))

    @staticmethod
    def mongo_jobs(dbname):
        return MongoJobs.new_from_connection_str(
            TempMongo.connection_string(dbname))

    def db_up(self):
        try:
            self.mongo_jobs('__test_db')
            return True
        except:  # XXX: don't know what exceptions to put here
            return False


def test_parse_url():
    uris = [
        'mongo://hyperopt:foobar@127.0.0.1:27017/hyperoptdb/jobs',
        'mongo://hyperopt:foobar@127.0.0.1:27017/hyperoptdb/jobs?authSource=db1'
    ]
    
    expected = [
        ('mongo', 'hyperopt', 'foobar', '127.0.0.1', 27017, 'hyperoptdb', 'jobs', None),
        ('mongo', 'hyperopt', 'foobar', '127.0.0.1', 27017, 'hyperoptdb', 'jobs', 'db1')
    ]
    
    for i, uri in enumerate(uris):
        assert parse_url(uri) == expected[i] 


# -- If we can't create a TempMongo instance, then
#    simply print what happened,
try:
    with TempMongo() as temp_mongo:
        pass
except OSError as e:
    print(e, file=sys.stderr)
    print(("Failed to create a TempMongo context,"
           " skipping all mongo tests."), file=sys.stderr)
    if "such file" in str(e):
        print("Hint: is mongod executable on path?", file=sys.stderr)
    raise nose.SkipTest()


class TestMongoTrials(hyperopt.tests.test_base.TestTrials):

    def setUp(self):
        self.temp_mongo = TempMongo()
        self.temp_mongo.__enter__()
        self.trials = MongoTrials(
            self.temp_mongo.connection_string('foo'),
            exp_key=None)

    def tearDown(self, *args):
        self.temp_mongo.__exit__(*args)


def with_mongo_trials(f, exp_key=None):
    def wrapper():
        with TempMongo() as temp_mongo:
            trials = MongoTrials(temp_mongo.connection_string('foo'),
                                 exp_key=exp_key)
            print('Length of trials: ', len(trials.results))
            f(trials)
    wrapper.__name__ = f.__name__
    return wrapper


def _worker_thread_fn(host_id, n_jobs, timeout, dbname='foo', logfilename=None):
    mw = MongoWorker(
        mj=TempMongo.mongo_jobs(dbname),
        logfilename=logfilename,
        workdir="mongoexp_test_dir",
    )
    try:
        while n_jobs:
            mw.run_one(host_id, timeout, erase_created_workdir=True)
            print('worker: %s ran job' % str(host_id))
            n_jobs -= 1
    except ReserveTimeout:
        print('worker timed out:', host_id)
        pass


def with_worker_threads(n_threads, dbname='foo',
                        n_jobs=sys.maxsize, timeout=10.0):
    """
    Decorator that will run a test with some MongoWorker threads in flight
    """
    def newth(ii):
        return threading.Thread(
            target=_worker_thread_fn,
            args=(('hostname', ii), n_jobs, timeout, dbname))

    def deco(f):
        def wrapper(*args, **kwargs):
            # --start some threads
            threads = list(map(newth, list(range(n_threads))))
            [th.start() for th in threads]
            try:
                return f(*args, **kwargs)
            finally:
                [th.join() for th in threads]
        wrapper.__name__ = f.__name__  # -- nose requires test in name
        return wrapper
    return deco


@with_mongo_trials
def test_with_temp_mongo(trials):
    pass  # -- just verify that the decorator can run


@with_mongo_trials
def test_new_trial_ids(trials):
    a = trials.new_trial_ids(1)
    b = trials.new_trial_ids(2)
    c = trials.new_trial_ids(3)

    assert len(a) == 1
    assert len(b) == 2
    assert len(c) == 3
    s = set()
    s.update(a)
    s.update(b)
    s.update(c)
    assert len(s) == 6


@with_mongo_trials
def test_attachments(trials):
    blob = b'abcde'
    assert 'aname' not in trials.attachments
    trials.attachments['aname'] = blob
    assert 'aname' in trials.attachments
    assert trials.attachments[u'aname'] == blob
    assert trials.attachments['aname'] == blob

    blob2 = b'zzz'
    trials.attachments['aname'] = blob2
    assert 'aname' in trials.attachments
    assert trials.attachments[u'aname'] == blob2
    assert trials.attachments['aname'] == blob2

    del trials.attachments['aname']
    assert 'aname' not in trials.attachments


@with_mongo_trials
def test_delete_all_on_attachments(trials):
    trials.attachments['aname'] = 'a'
    trials.attachments['aname2'] = 'b'
    assert 'aname2' in trials.attachments
    trials.delete_all()
    assert 'aname' not in trials.attachments
    assert 'aname2' not in trials.attachments


def test_handles_are_independent():
    with TempMongo() as tm:
        t1 = tm.mongo_jobs('t1')
        t2 = tm.mongo_jobs('t2')
        assert len(t1) == 0
        assert len(t2) == 0

        # test that inserting into t1 doesn't affect t2
        t1.insert({'a': 7})
        assert len(t1) == 1
        assert len(t2) == 0


def passthrough(x):
    assert os.path.split(os.getcwd()).count("mongoexp_test_dir") == 1, "cwd is %s" % os.getcwd()
    return x


class TestExperimentWithThreads(unittest.TestCase):

    @staticmethod
    def worker_thread_fn(host_id, n_jobs, timeout):
        mw = MongoWorker(
            mj=TempMongo.mongo_jobs('foodb'),
            logfilename=None,
            workdir="mongoexp_test_dir")
        while n_jobs:
            mw.run_one(host_id, timeout, erase_created_workdir=True)
            print('worker: %s ran job' % str(host_id))
            n_jobs -= 1

    @staticmethod
    def fmin_thread_fn(space, trials, max_evals, seed):
        fmin(
            fn=passthrough,
            space=space,
            algo=rand.suggest,
            trials=trials,
            rstate=np.random.RandomState(seed),
            max_evals=max_evals,
            return_argmin=False)

    def test_seeds_AAB(self):
        # launch 3 simultaneous experiments with seeds A, A, B.
        # Verify all experiments run to completion.
        # Verify first two experiments run identically.
        # Verify third experiment runs differently.

        exp_keys = ['A0', 'A1', 'B']
        seeds = [1, 1, 2]
        n_workers = 2
        jobs_per_thread = 6
        # -- total jobs = 2 * 6 = 12
        # -- divided by 3 experiments: 4 jobs per fmin
        max_evals = (n_workers * jobs_per_thread) // len(exp_keys)

        # -- should not matter which domain is used here
        domain = gauss_wave2()

        pickle.dumps(domain.expr)
        pickle.dumps(passthrough)

        worker_threads = [
            threading.Thread(
                target=TestExperimentWithThreads.worker_thread_fn,
                args=(('hostname', ii), jobs_per_thread, 30.0))
            for ii in range(n_workers)]

        with TempMongo() as tm:
            mj = tm.mongo_jobs('foodb')
            print(mj)
            trials_list = [
                MongoTrials(tm.connection_string('foodb'), key)
                for key in exp_keys]

            fmin_threads = [
                threading.Thread(
                    target=TestExperimentWithThreads.fmin_thread_fn,
                    args=(domain.expr, trials, max_evals, seed))
                for seed, trials in zip(seeds, trials_list)]

            try:
                [th.start() for th in worker_threads + fmin_threads]
            finally:
                print('joining worker threads...')
                [th.join() for th in worker_threads + fmin_threads]

            # -- not using an exp_key gives a handle to all the trials
            #    in foodb
            all_trials = MongoTrials(tm.connection_string('foodb'))
            self.assertEqual(len(all_trials), n_workers * jobs_per_thread)

            # Verify that the fmin calls terminated correctly:
            for trials in trials_list:
                self.assertEqual(
                    trials.count_by_state_synced(JOB_STATE_DONE),
                    max_evals)
                self.assertEqual(
                    trials.count_by_state_unsynced(JOB_STATE_DONE),
                    max_evals)
                self.assertEqual(len(trials), max_evals)

            # Verify that the first two experiments match.
            # (Do these need sorting by trial id?)
            trials_A0, trials_A1, trials_B0 = trials_list
            self.assertEqual(
                [t['misc']['vals'] for t in trials_A0.trials],
                [t['misc']['vals'] for t in trials_A1.trials])

            # Verify that the last experiment does not match.
            # (Do these need sorting by trial id?)
            self.assertNotEqual(
                [t['misc']['vals'] for t in trials_A0.trials],
                [t['misc']['vals'] for t in trials_B0.trials])


class FakeOptions(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# -- assert that the test raises a ReserveTimeout within 5 seconds
@nose.tools.timed(10.0)  # XXX:  this needs a suspiciously long timeout
@nose.tools.raises(ReserveTimeout)
@with_mongo_trials
def test_main_worker(trials):
    options = FakeOptions(
        max_jobs=1,
        # XXX: sync this with TempMongo
        mongo=as_mongo_str('localhost:22334/foodb'),
        reserve_timeout=1,
        poll_interval=.5,
        workdir=None,
        exp_key='foo',
        last_job_timeout=None,
    )
    # -- check that it runs
    #    and that the reserve timeout is respected
    main_worker_helper(options, ())
