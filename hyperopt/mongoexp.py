"""
Mongo-based Experiment driver
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "James Bergstra <pylearn-dev@googlegroups.com>"

import copy
import cPickle
import datetime
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urlparse

import numpy
import pymongo
import gridfs
from bson.son import SON

import base
import utils

logger = logging.getLogger(__name__)

# Proxy that could be factored out
# if we also want to use CouchDB
# and JobmanDB classes with this interface
class OperationFailure (Exception):
    def __init__(self, arg):
        self.arg = arg

    def __str__(self):
        return str(self.arg)


def read_pw():
    username = 'hyperopt'
    password = open(os.path.join(os.getenv('HOME'), ".hyperopt")).read()[:-1]
    return dict(
            username=username,
            password=password)


def authenticate_for_db(db):
    d = read_pw()
    db.authenticate(d['username'], d['password'])

##
# Code stolen from Jobman
#
def parse_url(url, pwfile=None):
    """Unpacks a url of the form protocol://[username[:pw]]@hostname[:port]/db/collection

    :rtype: tuple of strings
    :returns: protocol, username, password, hostname, port, dbname, collection

    :note: If the password is not given in the url but the username is, then this function
    will read the password from file by calling ``open(pwfile).read()[:-1]``

    """

    protocol=url[:url.find(':')]
    ftp_url='ftp'+url[url.find(':'):]

    #parse the string as if it were an ftp address
    tmp = urlparse.urlparse(ftp_url)

    logger.info( 'PROTOCOL %s'% protocol)
    logger.info( 'USERNAME %s'% tmp.username)
    logger.info( 'HOSTNAME %s'% tmp.hostname)
    logger.info( 'PORT %s'% tmp.port)
    logger.info( 'PATH %s'% tmp.path)
    try:
        _, dbname, collection = tmp.path.split('/')
    except:
        print >> sys.stderr, "Failed to parse '%s'"%(str(tmp.path))
        raise
    logger.info( 'DB %s'% dbname)
    logger.info( 'COLLECTION %s'% collection)

    if tmp.password is None:
        if (tmp.username is not None) and pwfile:
            password = open(pwfile).read()[:-1]
        else:
            password = None
    else:
        password = tmp.password
    logger.info( 'PASS %s'% password)

    return protocol, tmp.username, password, tmp.hostname, tmp.port, dbname, collection


def connection_with_tunnel(host='localhost',
            auth_dbname='admin', port=27017, 
            ssh=False, user='hyperopt', pw=None):
        if ssh:
            local_port=numpy.random.randint(low=27500, high=28000)
            # forward from local to remote machine
            ssh_tunnel = subprocess.Popen(['ssh', '-NTf', '-L', '%i:%s:%i'%(local_port,
                '127.0.0.1', port), host], 
                    #stdin=subprocess.PIPE,
                    #stdout=subprocess.PIPE,
                    #stderr=subprocess.PIPE,
                    )
            time.sleep(.5) # give the subprocess time to set up
            connection = pymongo.Connection('127.0.0.1', local_port, document_class=SON)
        else:
            connection = pymongo.Connection(host, port, document_class=SON)
            if user:
                if user == 'hyperopt':
                    authenticate_for_db(connection[auth_dbname])
                else:
                    raise NotImplementedError()
            ssh_tunnel=None

        return connection, ssh_tunnel


def connection_from_string(s):
    protocol, user, pw, host, port, db, collection = parse_url(s)
    if protocol == 'mongo':
        ssh=False
    elif protocol in ('mongo+ssh', 'ssh+mongo'):
        ssh=True
    else:
        raise ValueError('unrecognized protocol for MongoJobs', protocol)
    connection, tunnel = connection_with_tunnel(
            ssh=ssh,
            user=user,
            pw=pw,
            host=host,
            port=port,
            )
    return connection, tunnel, connection[db], connection[db][collection]


def coarse_utcnow():
    # MongoDB stores only to the nearest millisecond
    # This is mentioned in a footnote here:
    # http://api.mongodb.org/python/1.9%2B/api/bson/son.html#dt
    now = datetime.datetime.utcnow()
    microsec = (now.microsecond//10**3)*(10**3)
    return datetime.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second, microsec)

class MongoJobs(object):
    # Interface to a Jobs database structured like this
    #
    # Collections:
    #
    # db.jobs - structured {'argd', 'cmd', 'owner', 'book_time', 'refresh_time', 'state',
    #                       'error', 'result'}
    #    This is the collection that the worker nodes write to
    #
    # db.gfs - file storage via gridFS for all collections
    #
    def __init__(self, db, jobs, gfs, conn, tunnel, argd_name):
        self.db = db
        self.jobs = jobs
        self.gfs = gfs
        self.conn=conn
        self.tunnel=tunnel
        self.argd_name = argd_name

    @classmethod
    def alloc(cls, dbname, host='localhost',
            auth_dbname='admin', port=27017,
            jobs_coll='jobs', gfs_coll='fs', ssh=False, user=None, pw=None):
        connection, tunnel = connection_with_tunnel(
                host, auth_dbname, port, ssh, user, pw)
        db = connection[dbname]
        gfs = gridfs.GridFS(db, collection=gfs_coll)
        return cls(db, db[jobs_coll], gfs, connection, tunnel)

    @classmethod
    def new_from_connection_str(cls, conn_str, gfs_coll='fs', argd_name='spec'):
        connection, tunnel, db, coll = connection_from_string(conn_str)
        gfs = gridfs.GridFS(db, collection=gfs_coll)
        return cls(db, coll, gfs, connection, tunnel, argd_name)

    def __iter__(self):
        return self.jobs.find()

    def __len__(self):
        try:
            return self.jobs.count()
        except:
            return 0

    def jobs_complete(self, cursor=False):
        c = self.jobs.find(spec=dict(state=2))
        return c if cursor else list(c)
    def jobs_error(self, cursor=False):
        c = self.jobs.find(spec=dict(state=3))
        return c if cursor else list(c)
    def jobs_running(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(spec=dict(state=1)))
        #TODO: mark some as MIA
        rval = [r for r in rval if not r.get('MIA', False)]
        return rval
    def jobs_dead(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(spec=dict(state=1)))
        #TODO: mark some as MIA
        rval = [r for r in rval if r.get('MIA', False)]
        return rval
    def jobs_queued(self, cursor=False):
        c = self.jobs.find(spec=dict(state=0))
        return c if cursor else list(c)

    def insert(self, job, safe=True):
        """Return a job dictionary by inserting the job dict into the database"""
        try:
            cpy = copy.deepcopy(job)
            # this call adds an _id field to cpy
            _id = self.jobs.insert(cpy, safe=safe, check_keys=True)
            # so now we return the dict with the _id field
            assert _id == cpy['_id']
            return cpy
        except pymongo.errors.OperationFailure, e:
            raise OperationFailure(e)

    def delete(self, job, safe=True):
        """Delete job[s]"""
        try:
            self.jobs.remove(job, safe=safe)
        except pymongo.errors.OperationFailure, e:
            raise OperationFailure(e)

    def delete_all(self, cond={}, safe=True):
        """Delete all jobs and attachments"""
        try:
            for d in self.jobs.find(spec=cond, fields=['_id', '_attachments']):
                for name, file_id in d.get('_attachments',[]):
                    self.gfs.delete(file_id)
                logger.info('deleting job %s' % d['_id'])
                self.jobs.remove(d, safe=safe)
        except pymongo.errors.OperationFailure, e:
            raise OperationFailure(e)
    def delete_all_error_jobs(self, safe=True):
        return self.delete_all(cond={'state':3}, safe=safe)

    def reserve(self, host_id, cond=None):
        now = coarse_utcnow()
        if cond is None:
            cond = {}
        else:
            cond = copy.copy(cond) #copy is important, will be modified, but only the top-level

        if 'owner' not in cond:
            cond['owner'] = None

        if cond['owner'] is not None:
            raise ValueError('refusing to reserve owned job')
        try:
            self.jobs.update( 
                cond, 
                {'$set': 
                    {'owner':host_id,
                     'book_time':now,
                     'state':1,
                     'refresh_time':now,
                     }
                 },
                safe=True,
                upsert=False,
                multi=False,)
        except pymongo.errors.OperationFailure, e:
            logger.error('Error during reserve_job: %s'%str(e))
            return None
        cond['owner'] = host_id
        cond['book_time'] = now
        return self.jobs.find_one(cond)

    def refresh(self, job, safe=False):
        self.update(job, dict(refresh_time=coarse_utcnow()), safe=False)

    def update(self, job, dct, safe=True):
        """Return union of job and dct, after making sure that dct has been added to job in db.

        This function does not modify either `job` or `dct`.

        safe=True means error-checking is done. safe=False means this function will succeed
        regardless of what happens with the db.
        """
        dct = copy.deepcopy(dct)
        if '_id' in dct:
            raise ValueError('cannot update the _id field')
        if 'version' in dct:
            raise ValueError('cannot update the version field')
        if '_id' not in job:
            raise ValueError('job must have an "_id" key to be updated')

        if 'version' in job:
            job_query = dict(_id=job['_id'], version=job['version'])
            dct['version'] = job['version']+1
        else:
            job_query = dict(_id=job['_id'])
            dct['version'] = 1
        try:
            # warning - if job matches nothing then this function succeeds
            # N.B. this matches *at most* one entry, and possibly zero
            self.jobs.update( 
                    job_query,
                    {'$set': dct},
                    safe=True,
                    upsert=False,
                    multi=False,)
        except pymongo.errors.OperationFailure, e:
            # translate pymongo failure into generic failure
            raise OperationFailure(e)

        # update job in-place to match what happened on the server side
        job.update(dct)

        if safe:
            server_job = self.jobs.find_one(dict(_id=job['_id'], version=job['version']))
            if server_job is None:
                raise OperationFailure('updated job not found in collection: %s'%str(job))
            elif server_job != job:
                if 0:# This is all commented out because it is tripping on the fact that
                    # str('a') != unicode('a').
                    # TODO: eliminate false alarms and catch real ones
                    mismatching_keys = []
                    for k,v in server_job.items():
                        if k in job:
                            if job[k] != v:
                                mismatching_keys.append((k, v, job[k]))
                        else:
                            mismatching_keys.append((k, v, '<missing>'))
                    for k,v in job.items():
                        if k not in server_job:
                            mismatching_keys.append((k, '<missing>', v))

                    raise OperationFailure('local and server job documents are out of sync: %s'%
                            repr((job, server_job, mismatching_keys)))
        return job

    def attachment_names(self, job):
        return [a[0] for a in job.get('_attachments', [])]

    def add_attachment(self, job, blob, name):
        """Attach potentially large data string `blob` to `job` by name `name`

        Returns None
        """

        # If there is already a file with the given name for this job, then we will delete it
        # after writing the new file
        attachments = job.get('_attachments', [])
        old_name_idx = -1
        for i,a in enumerate(attachments):
            if a[0] == name:
                old_name_idx == i
                old_file_id = a[1]
                break

        # the filename is set to something so that fs.list() will display the file
        new_file_id = self.gfs.put(blob, filename='%s_%s'%(job['_id'],name))
        #print "stored blob", new_file_id

        if old_name_idx >= 0:
            new_attachments = attachments
            new_attachments[old_name_idx] = (name, new_file_id)
        else:
            new_attachments = attachments + [(name, new_file_id)]

        try:
            leak=False
            job = self.update(job, {'_attachments':new_attachments})
            if old_name_idx >= 0:
                leak=True
                self.gfs.delete(old_file_id)
                leak=False
        except:
            if leak:
                logger.warning("Leak during attach_blob: old_file_id=%s"% (old_file_id,))
            raise
        #return new_file_id

    def get_attachment(self, job, name):
        """Retrieve data attached to `job` by `attach_blob`.

        Raises KeyError if `name` does not correspond to an attached blob.

        Returns the blob as a string.
        """
        attachments = job.get('_attachments', [])
        file_ids = [a[1] for a in attachments if a[0] == name]
        if not file_ids:
            raise OperationFailure('Attachment not found: %s' % name)
        assert len(file_ids) < 2
        return self.gfs.get(file_ids[0]).read()

    def delete_attachment(self, job, name):
        attachments = job.get('_attachments', [])
        file_id = None
        for i,a in enumerate(attachments):
            if a[0] == name:
                file_id = a[1]
                break
        if file_id is None:
            raise OperationFailure('Attachment not found: %s' % name)
        #print "Deleting", file_id
        del attachments[i]
        self.update(job, {'_attachments':attachments})
        self.gfs.delete(file_id)


class MongoExperiment(base.Experiment):
    """
    This experiment uses a Mongo collection to store
    - self.trials
    - self.results

    """
    def __init__(self, bandit_json, bandit_algo_json, mongo_handle, workdir,
            exp_key=None, poll_interval_secs = 10):
        # don't call base Experiment because it tries to set trials = []
        # base.Experiment.__init__(self, bandit, bandit_algo)
        self.bandit_json = bandit_json
        self.bandit = utils.json_call(bandit_json)
        self.bandit_algo = utils.json_call(bandit_algo_json)
        self.bandit_algo.set_bandit(self.bandit)
        self.workdir = workdir
        if isinstance(mongo_handle, str):
            self.mongo_handle = MongoJobs.new_from_connection_str(
                    mongo_handle,
                    argd_name='spec')
        else:
            self.mongo_handle = mongo_handle
        config = self.mongo_handle.db.config.find_one()
        if config is None:
            logger.info('inserting config document')
            config = {'workdir': self.workdir}
            self.mongo_handle.db.config.insert(config)
        else:
            logger.info('found config document %s' % str(config))
        self.poll_interval_secs = poll_interval_secs
        self.min_queue_len = 1 # can be changed at any time
        if exp_key is None:
            self.exp_key = (bandit_json, bandit_algo_json)
        else:
            self.exp_key = exp_key

    def __get_trials(self):
        # TODO: this query can be done much more efficiently
        query = {'exp_key': self.exp_key, 'result': {'$ne': None}}
        all_jobs = list(self.mongo_handle.jobs.find(query))
        id_jobs = [(j['_id'], j) for j in all_jobs]
        id_jobs.sort()
        return [id_job[1]['spec'] for id_job in id_jobs]
    trials = property(__get_trials)

    def __get_results(self):
        # TODO: this query can be done much more efficiently
        query = {'exp_key': self.exp_key, 'result': {'$ne': None}}
        all_jobs = list(self.mongo_handle.jobs.find(query))
        #logger.info('all jobs: %s' % str(all_jobs))
        id_jobs = [(j['_id'], j) for j in all_jobs]
        id_jobs.sort()
        return [id_job[1]['result'] for id_job in id_jobs]
    results = property(__get_results)

    def queue_extend(self, trial_specs):
        rval = []
        cmd = ('bandit_json evaluate', self.bandit_json)
        for spec in trial_specs:
            to_insert = dict(
                    state=0,
                    exp_key=self.exp_key,
                    cmd=cmd,
                    owner=None,
                    spec=spec,
                    result=None,
                    version=0,
                    )
            rval.append(self.mongo_handle.jobs.insert(to_insert, safe=True))
        return rval

    def queue_len(self):
        # TODO: consider searching by SON rather than dict
        query = dict(state=0, exp_key=self.exp_key)
        rval = self.mongo_handle.jobs.find(query).count()
        logger.info('Queue len: %i' % rval)
        return rval

    def run(self, N):
        bandit = self.bandit
        algo = self.bandit_algo

        n_queued = 0

        while n_queued < N:
            while self.queue_len() < self.min_queue_len:
                suggestions = algo.suggest(
                        self.trials, self.Ys(), self.Ys_status(), 1)
                logger.info('algo suggested trial: %s' % str(suggestions[0]))
                new_suggestions = []
                for spec in suggestions:
                    spec_query = self.mongo_handle.jobs.find(dict(spec=spec))
                    if spec_query.count():
                        spec_matches = list(spec_query)
                        assert len(spec_matches) == 1
                        logger.info('Skipping duplicate trial')
                    else:
                        new_suggestions.append(spec)
                new_suggestions = self.queue_extend(new_suggestions)
                n_queued += len(new_suggestions)
            time.sleep(self.poll_interval_secs)

    @classmethod
    def main_search(cls, argv):
        mongo_str = argv[0]
        bandit_json = argv[1]
        bandit_algo_json = argv[2]
        workdir = argv[3]
        self = cls(bandit_json, bandit_algo_json, mongo_str, workdir)
        self.run(sys.maxint)


class Shutdown(Exception):
    pass


class CtrlObj(object):
    def __init__(self, read_only=False, read_only_id=None, **kwargs):
        self.current_job = None
        self.__dict__.update(kwargs)
        self.read_only=read_only
        self.read_only_id=read_only_id

    def debug(self, *args, **kwargs):
        return logger.debug(*args, **kwargs)
    def info(self, *args, **kwargs):
        return logger.info(*args, **kwargs)
    def warn(self, *args, **kwargs):
        return logger.warn(*args, **kwargs)
    def error(self, *args, **kwargs): 
        return logger.error(*args, **kwargs)
    def checkpoint(self, result=None):
        if not self.read_only:
            self.jobs.refresh(self.current_job)
            if rval is not None:
                return self.jobs.update(self.current_job, dict(result=result))


def exec_import(cmd_module, cmd):
    exec('import %s; worker_fn = %s' % (cmd_module, cmd))
    return worker_fn

def main_worker():
    # usage to launch worker:
    #     hyperopt-worker mongo://...
    # usage to launch worker loop:
    #     hyperopt-worker mongo://... N
    argv = sys.argv

    try:
        script, mongo_str, N = argv
    except:
        try:
            script, mongo_str = argv
            N = 1
        except:
            raise Exception('TODO USAGE too many or too few arguments', argv)

    assert 'hyperopt-worker' in script  # not essential, just debugging here
    N = int(N)

    if N != 1:
        if N < 0:
            MAX_CONSECUTIVE_FAILURES = abs(N)
            N = - 1
        else:
            MAX_CONSECUTIVE_FAILURES = 10
            N = N

        def sighandler_shutdown(signum, frame):
            logger.info('Caught signal %i, shutting down.' % signum)
            raise Shutdown(signum)
        signal.signal(signal.SIGINT, sighandler_shutdown)
        signal.signal(signal.SIGHUP, sighandler_shutdown)
        signal.signal(signal.SIGTERM, sighandler_shutdown)
        proc = None
        consecutive_exceptions = 0
        while N and consecutive_exceptions < MAX_CONSECUTIVE_FAILURES:
            try:
                # recursive Popen, dropping N from the argv
                # By using another process to run this job
                # we protect ourselves from memory leaks, bad cleanup
                # and other annoying details.
                # The tradeoff is that a large dataset must be reloaded once for
                # each subprocess.
                proc = subprocess.Popen(sys.argv[:-1])
                retcode = proc.wait()
                proc = None
            except Shutdown, e:
                #this is the normal way to stop the infinite loop (if originally N=-1)
                if proc:
                    #proc.terminate() is only available as of 2.6
                    os.kill(proc.pid, signal.SIGTERM)
                    return proc.wait()
                else:
                    return 0

            if retcode != 0:
                consecutive_exceptions += 1
            else:
                consecutive_exceptions = 0
            N -= 1
        logger.info("exiting with N=%i after %i consecutive exceptions" %(
            N, consecutive_exceptions))
    else:
        mongojobs = MongoJobs.new_from_connection_str(mongo_str,
                argd_name='spec')
        job = None
        while job is None:
            job = mongojobs.reserve(
                    host_id = '%s:%i'%(socket.gethostname(), os.getpid()),
                    )
            if not job:
                logger.info('no job found, sleeping for up to 5 seconds')
                time.sleep(numpy.random.rand() * 5)

        spec = copy.deepcopy(job['spec']) # don't let the cmd mess up our trial object

        ctrl = CtrlObj(
                jobs=mongojobs,
                read_only=False,
                read_only_id=None,
                )
        ctrl.current_job = job
        config = mongojobs.db.config.find_one()
        workdir = os.path.join(config['workdir'], str(job['_id']))
        os.makedirs(workdir)
        os.chdir(workdir)
        cmd_protocol = job['cmd'][0]
        try:
            if cmd_protocol == 'cpickled fn':
                worker_fn = cPickle.loads(job['cmd'][1])
                result = worker_fn(spec, ctrl)
            elif cmd_protocol == 'call evaluate':
                bandit = cPickle.loads(job['cmd'][1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'token_load':
                cmd_toks = cmd.split('.')
                cmd_module = '.'.join(cmd_toks[:-1])
                worker_fn = exec_import(cmd_module, cmd)
                result = worker_fn(spec, ctrl)
            elif cmd_protocol == 'bandit_json evaluate':
                bandit = utils.json_call(job['cmd'][1])
                worker_fn = bandit.evaluate
                result = worker_fn(spec, ctrl)
            else:
                raise ValueError('Unrecognized cmd protocol', cmd_protocol)
        except Exception, e:
            #TODO: save exception to database, but if this fails, then at least raise the original
            # traceback properly
            mongojobs.update(job,
                    {'state': 3,
                    'error': (str(type(e)), str(e))},
                    safe=True)
            raise
        mongojobs.update(job, {'state': 2, 'result': result}, safe=True)
