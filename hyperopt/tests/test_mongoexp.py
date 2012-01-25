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

from hyperopt.utils import json_call
from hyperopt.mongoexp import MongoExperiment
from hyperopt.mongoexp import MongoWorker
from hyperopt.mongoexp import as_mongo_str
from hyperopt.mongoexp import MongoJobs
from hyperopt.mongoexp import OperationFailure
from hyperopt.bandits import TwoArms, GaussWave2
from hyperopt import bandit_algos
from hyperopt.theano_gp import HGP
from hyperopt.theano_bandit_algos import TheanoRandom


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
                    cwd=self.workdir,
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

    @classmethod
    def mongo_jobs(self, name):
        return MongoJobs.new_from_connection_str(
                as_mongo_str('localhost:22334/%s' % name) + '/jobs')

    def db_up(self):
        try:
            self.mongo_jobs('__test_db')
            return True
        except:  # XXX: don't know what exceptions to put here
            return False


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
            bandit = json_call('hyperopt.bandits.GaussWave2')
            bandit_algo = json_call(algo_json, args=(bandit,))
            exp = MongoExperiment(
                bandit_algo=bandit_algo,
                mongo_handle=tm.mongo_jobs('foodb'),
                workdir=tm.workdir,
                exp_key='exp_key',
                poll_interval_secs=1.0,
                cmd=('asdf', None))
            exp_str = cPickle.dumps(exp)
            cpy = cPickle.loads(exp_str)


def test_multiple_mongo_exps_with_threads():
    def worker(host_id, n_jobs, timeout, exp_key):
        mw = MongoWorker(
            mj=TempMongo.mongo_jobs('foodb'),
            exp_key=exp_key
            )
        t0 = time.time()
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
            'hyperopt.dbn.Dummy_DBN_Base',):
        with TempMongo() as tm:
            mj = tm.mongo_jobs('foodb')
            mj.conn.drop_database('foodb')  #need to clean stuff out! should this be in a TempMongo method?
            assert len(mj) == 0, len(mj)
            bandit = json_call(bandit_json)
            exp = MongoExperiment(
                bandit_algo=HGP(bandit),
                mongo_handle=tm.mongo_jobs('foodb'),
                workdir=tm.workdir,
                exp_key='exp_key',
                poll_interval_secs=1.0,
                cmd=('bandit_json evaluate', bandit_json))
            exp.ddoc_init()
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

            #print exp.trials
            #print exp.results
            assert len(exp.trials) == n_trials, 'trials failure %d %d ' % (len(exp.trials) , n_trials)
            assert len(exp.results) == n_trials, 'results failure %d %d ' % (len(exp.results) , n_trials)

            exp_str = cPickle.dumps(exp)  #is anything done with this exp_str?


class TestLock(unittest.TestCase):
    def setUp(self):
        self.ctxt = TempMongo().__enter__()
        try:
            self.a = MongoExperiment(
                bandit_algo=bandit_algos.Random(TwoArms()),
                mongo_handle=self.ctxt.mongo_jobs('foodb'),
                workdir=self.ctxt.workdir,
                exp_key='exp_key',
                poll_interval_secs=1.0,
                cmd=())
            # create a second experiment with same key
            self.b = MongoExperiment(
                bandit_algo=bandit_algos.Random(GaussWave2()),
                mongo_handle=self.ctxt.mongo_jobs('foodb'),
                workdir=self.ctxt.workdir,
                exp_key='exp_key',
                poll_interval_secs=1.0,
                cmd=())
            self.c = MongoExperiment(
                bandit_algo=bandit_algos.Random(GaussWave2()),
                mongo_handle=self.ctxt.mongo_jobs('foodb'),
                workdir=self.ctxt.workdir,
                exp_key='exp_key_different',
                poll_interval_secs=1.0,
                cmd=())
            self.a.ddoc_init()
            self.c.ddoc_init()
        except:
            self.ctxt.__exit__(None)
            raise

    def tearDown(self):
        self.ctxt.__exit__(None)

    def test_lock_relock(self):
        with self.a.exclusive_access() as foo:
            self.assertRaises(OperationFailure, self.b.ddoc_lock)

        with self.b.exclusive_access() as foo:
            # test that c can still be acquired with b locked
            self.assertRaises(OperationFailure, self.a.ddoc_lock)
            with self.c.exclusive_access() as bar:
                self.assertRaises(OperationFailure, self.a.ddoc_lock)

        with self.a.exclusive_access() as foo:
            with self.a.exclusive_access() as foo2:
                with self.a.exclusive_access() as foo3:
                    self.assertRaises(OperationFailure, self.b.ddoc_lock)
                    assert self.a._locks == 3
                assert self.a._locks == 2
            self.assertRaises(OperationFailure, self.b.ddoc_lock)
            assert self.a._locks == 1
        assert self.a._locks == 0

    def test_clear_requires_lock(self):
        with self.a.exclusive_access() as foo:
            self.assertRaises(OperationFailure, self.b.clear_from_db)

# XXX: test blocking behaviour

# XXX: Test clear_db only removes things matching exp_key


# XXX: test that multiple experiments can run simultaneously on the same
#      jobs table using threads


# XXX: find old mongoexp unit tests and put them in here
