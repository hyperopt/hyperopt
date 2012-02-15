if 0:
    # This code prints out the offending object when pickling fails
    import copy_reg
    orig_reduce_ex = copy_reg._reduce_ex
    def my_reduce_ex(self, proto):
        try:
            return orig_reduce_ex(self, proto)
        except:
            print 'PICKLING FAILED', self
            raise
    copy_reg._reduce_ex = my_reduce_ex

import cPickle
import os
import signal
import subprocess
import sys
import threading
import time
import unittest

import nose

from hyperopt import Experiment
from hyperopt import Random
from hyperopt.base import JOB_STATE_DONE
from hyperopt.utils import json_call
from hyperopt.mongoexp import MongoTrials
from hyperopt.mongoexp import MongoWorker
from hyperopt.mongoexp import as_mongo_str
from hyperopt.mongoexp import _MongoJobs

from hyperopt.mongoexp import OperationFailure
from hyperopt.bandits import TwoArms, GaussWave2


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
            proc_args = [ "mongod",
                        "--dbpath=%s/db" % self.workdir,
                        "--nojournal",
                         "--noprealloc",
                        "--port=22334"]
            #print "starting mongod", proc_args
            self.mongo_proc = subprocess.Popen(
                    proc_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.workdir, # this prevented mongod assertion fail 
                    )
            try:
                interval = .125
                while interval <= 2:
                    if interval > .125:
                        print "Waiting for mongo to come up"
                    time.sleep(interval)
                    interval *= 2
                    if  self.db_up():
                        break
                if self.db_up():
                    return self
                else:
                    try:
                        os.kill(self.mongo_proc.pid, signal.SIGTERM)
                    except OSError:
                        pass # if it crashed there is no such process
                    out, err = self.mongo_proc.communicate()
                    print >> sys.stderr, out
                    print >> sys.stderr, err
                    raise RuntimeError('No database connection', proc_args)
            except Exception, e:
                try:
                    os.kill(self.mongo_proc.pid, signal.SIGTERM)
                except OSError:
                    pass # if it crashed there is no such process
                raise e

    def __exit__(self, *args):
        #print 'CLEANING UP MONGO ...'
        os.kill(self.mongo_proc.pid, signal.SIGTERM)
        self.mongo_proc.wait()
        subprocess.call(["rm", "-Rf", self.workdir])
        #print 'CLEANING UP MONGO DONE'

    @staticmethod
    def connection_string(dbname):
        return as_mongo_str('localhost:22334/%s' % dbname) + '/jobs'

    @staticmethod
    def mongo_jobs(dbname):
        return _MongoJobs.new_from_connection_str(
                TempMongo.connection_string(dbname))

    def db_up(self):
        try:
            self.mongo_jobs('__test_db')
            return True
        except:  # XXX: don't know what exceptions to put here
            return False


def with_mongo_trials(f):
    def wrapper():
        with TempMongo() as temp_mongo:
            trials = MongoTrials(temp_mongo.connection_string('foo'),
                    exp_key=None)
            f(trials)
    wrapper.__name__ = f.__name__
    return wrapper


@with_mongo_trials
def test_with_temp_mongo(trials):
    pass # -- just verify that the decorator can run

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


class TestExperimentWithThreads(unittest.TestCase):

    @staticmethod
    def worker_thread_fn(host_id, n_jobs, timeout):
        mw = MongoWorker(mj=TempMongo.mongo_jobs('foodb'))
        while n_jobs:
            mw.run_one(host_id, timeout)
            print 'worker: ran job'
            n_jobs -= 1

    def work(self, bandit):
        """
        Run a small experiment with several workers running in parallel
        using Python threads.
        """
        n_threads = 3
        jobs_per_thread = 2
        n_trials = n_threads * jobs_per_thread

        with TempMongo() as tm:
            mj = tm.mongo_jobs('foodb')
            trials = MongoTrials(tm.connection_string('foodb'))
            assert len(trials) == 0

            bandit_algo = Random(bandit)
            exp = Experiment(trials, bandit_algo, cmd=self.cmd)

            def newth():
                return threading.Thread(
                        target=self.worker_thread_fn,
                        args=(('hostname', 0), jobs_per_thread, 600.0))
            threads = [newth() for ii in range(n_threads)]
            [th.start() for th in threads]

            try:
                print 'running experiment'
                exp.run(n_trials, block_until_done=True)
            finally:
                print 'joining worker thread...'
                [th.join() for th in threads]


            assert trials.count_by_state_synced(JOB_STATE_DONE) == n_trials
            assert trials.count_by_state_unsynced(JOB_STATE_DONE) == n_trials
            assert len(trials) == n_trials, (
                'trials failure %d %d ' % (len(trials) , n_trials))
            assert len(trials.results) == n_trials, (
                'results failure %d %d ' % (len(trials.results) , n_trials))

    def test_bandit_json(self):
        self.cmd = ('bandit_json evaluate',
                'hyperopt.bandits.GaussWave2')
        self.work(GaussWave2())

# XXX: test each of the bandit calling protocols

# XXX: test blocking behaviour

# XXX: Test clear_db only removes things matching exp_key

# XXX: test that multiple experiments can run simultaneously on the same
#      jobs table using threads


# XXX: find old mongoexp unit tests and put them in here

def test_multiple_mongo_exps_with_threads():
    raise nose.SkipTest()
    def worker(host_id, n_jobs, timeout, exp_key):
        mw = MongoWorker(
            mj=TempMongo.mongo_jobs('foodb'),
            exp_key=exp_key
            )
        while n_jobs:
            mw.run_one(host_id, timeout)
            print 'worker: ran job'
            n_jobs -= 1

    bandit_jsons = ('hyperopt.bandits.GaussWave2',
                   'hyperopt.dbn.Dummy_DBN_Base',)

    with TempMongo() as tm:
        mj = tm.mongo_jobs('foodb')
        mj.conn.drop_database('foodb')  #need to clean stuff out! should this be in a TempMongo method?
        assert len(mj) == 0, len(mj)

        bandits = map(json_call, bandit_jsons)

        exps = []
        for bj, bandit in zip(bandit_jsons, bandits):
            exp = MongoExperiment(
                bandit_algo=TheanoRandom(bandit),
                mongo_handle=mj,
                workdir=tm.workdir,
                exp_key=bj,
                poll_interval_secs=1.0,
                cmd=('bandit_json evaluate', bj))
            print ('Initializing', exp.exp_key)
            exp.ddoc_init() 
            assert len(exp.mongo_handle) == 0
            exps.append(exp)

        n_trials = 5

        wthreads = []
        for bj, exp in zip(bandit_jsons, exps):
            wthread = threading.Thread(target=worker,
                                       args=(('worker_' + bj, 0), 
                                              n_trials, 660.0),
                                       kwargs={'exp_key': bj})
            wthread.start()  
            wthreads.append(wthread)

        try:
            print 'running experiments'
            for exp in exps:
                exp.run(n_trials)
            for exp in exps:
                exp.block_until_done()
        finally:
            print 'joining worker threads...'
            for wthread in wthreads:
                wthread.join()
    
        for bj, exp in zip(bandit_jsons, exps):
            assert len(exp.trials) == n_trials, '%d %d' % (len(exp.trials), 
                                                           n_trials)
            assert len(exp.results) == n_trials, '%d %d' % (len(exp.results),
                                                           n_trials)
            js = list(mj.db.jobs.find({'exp_key':bj}))
            hosts = set([_j['owner'][0] for _j in js])
            print 'hosts for exp %s: %s' % (exp.exp_key, ','.join(list(hosts)))
            assert hosts == set(['worker_' + bj])


