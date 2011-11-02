import cPickle
import os
import signal
import subprocess
import threading
import time
import unittest

import numpy

from hyperopt.base import Bandit, BanditAlgo, Experiment
from hyperopt.mongoexp import MongoExperiment, MongoWorker
from hyperopt.mongoexp import as_mongo_str
from hyperopt.mongoexp import MongoJobs
from hyperopt.bandits import TwoArms
from hyperopt import bandit_algos
from hyperopt.theano_gp import HGP


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
                    "--port=22334"]
            #print "starting mongod", proc_args
            self.mongo_proc = subprocess.Popen(proc_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
            time.sleep(.25) # wait for mongo to start up
            return self

    def __exit__(self, *args):
        #print 'CLEANING UP MONGO'
        os.kill(self.mongo_proc.pid, signal.SIGTERM)
        self.mongo_proc.wait()
        subprocess.call(["rm", "-Rf", self.workdir])

    @classmethod
    def mongo_jobs(self, name):
        return MongoJobs.new_from_connection_str(
                as_mongo_str('localhost:22334/%s' % name) + '/jobs')


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


def test_mongo_exp_is_picklable():
    algo_jsons = []
    algo_jsons.append('hyperopt.bandit_algos.Random')
    algo_jsons.append('hyperopt.theano_gp.HGP')
    algo_jsons.append('hyperopt.theano_gm.AdaptiveParzenGM')
    for algo_json in algo_jsons:
        with TempMongo() as tm:
            # this triggers if an old stale mongo is running
            assert len(TempMongo.mongo_jobs('foodb')) == 0
            print 'pickling MongoExperiment with', algo_json
            exp = MongoExperiment(
                bandit_json='hyperopt.bandits.GaussWave2',
                bandit_algo_json=algo_json,
                mongo_handle=tm.mongo_jobs('foodb'),
                workdir=tm.workdir,
                exp_key='exp_key',
                poll_interval_secs=1.0)
            # XXX: implement this: save_interval_secs=3.0)
            exp_str = cPickle.dumps(exp)
            cpy = cPickle.loads(exp_str)

def test_mongo_exp_with_threads():
    def worker(host_id, n_jobs, timeout):
        mw = MongoWorker(
            mj=TempMongo.mongo_jobs('foodb'),
            )
        t0 = time.time()
        while n_jobs:
            mw.run_one(host_id, timeout)
            print 'worker: ran job'
            n_jobs -= 1

    for bandit_json in ('hyperopt.bandits.GaussWave2',
            'hyperopt.dbn.Dummy_DBN_Base'):
        with TempMongo() as tm:
            assert len(TempMongo.mongo_jobs('foodb')) == 0
            exp = MongoExperiment(
                bandit_json=bandit_json,
                bandit_algo_json='hyperopt.theano_gp.HGP',
                mongo_handle=tm.mongo_jobs('foodb'),
                workdir=tm.workdir,
                exp_key='exp_key',
                poll_interval_secs=1.0)
            assert len(TempMongo.mongo_jobs('foodb')) == 0
            for asdf in exp.mongo_handle:
                print asdf
            assert len(exp.mongo_handle) == 0

            n_trials = 5
            exp.bandit_algo.n_startup_jobs = 3

            wthread = threading.Thread(target=worker,
                    args=(('hostname', 0), n_trials, 660.0))
            wthread.start()
            try:
                print 'running experiment'
                exp.run(n_trials, block_until_done=True)
            finally:
                print 'joining worker thread...'
                wthread.join()

            print exp.trials
            print exp.results
            assert len(exp.trials) == n_trials
            assert len(exp.results) == n_trials

            exp_str = cPickle.dumps(exp)


# XXX: test that multiple experiments can run simultaneously on the same
#      jobs table using threads


# XXX: find old mongoexp unit tests and put them in here
