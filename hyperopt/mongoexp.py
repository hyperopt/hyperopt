"""
Mongo-based Experiment driver and worker client
===============================================

Components involved:

- mongo
    e.g. mongod ...

- driver
    e.g. hyperopt-mongo-search mongo://address bandit_json bandit_algo_json

- worker
    e.g. hyperopt-mongo-worker --loop mongo://address


Mongo
=====

Mongo (daemon process mongod) is used for IPC between the driver and worker.
Configure it as you like, so that hyperopt-mongo-search can communicate with it.
I think there is some support in this file for an ssh+mongo connection type.

The experiment uses the following collections for IPC:

* jobs - documents of a standard form used to store suggested trials and their
    results.  These documents have keys:
    * spec : subdocument returned by bandit_algo.suggest
    * exp_key: an identifier of which driver suggested this trial
    * cmd: a tuple (protocol, ...) identifying bandit.evaluate
    * state: 0, 1, 2, 3 for job state (new, running, ok, fail)
    * owner: None for new jobs, (hostname, pid) for started jobs
    * book_time: time a job was reserved
    * refresh_time: last time the process running the job checked in
    * result: the subdocument returned by bandit.evaluate
    * error: for jobs of state 3, a reason for failure.
    * logs: a dict of sequences of strings received by ctrl object
        * info: info messages
        * warn: warning messages
        * error: error messages

* fs - a gridfs storage collection (used for pickling)

* drivers - documents describing drivers. These are used to prevent two drivers
    from using the same exp_key simultaneously, and to attach saved states.
    * exp_key
    * workdir: [optional] path where workers should chdir to
    Attachments:
        * pkl: [optional] saved state of experiment class
        * bandit_args_kwargs: [optional] pickled (clsname, args, kwargs) to
             reconstruct bandit in worker processes

The MongoJobs, MongoExperiment, and CtrlObj classes as well as the main_worker
method form the abstraction barrier around this database layout.


Driver
======

A driver directs an experiment, by calling a bandit_algo to suggest trial
points, and queuing them in mongo so that a worker can evaluate that trial
point.

The hyperopt-mongo-search script creates a single MongoExperiment instance, and
calls its run() method.


Saving and Resuming
-------------------

The command
"hyperopt-mongo-search bandit algo"
creates a new experiment or resumes an existing experiment.

The command
"hyperopt-mongo-search --exp-key=<EXPKEY>"
can only resume an existing experiment.

The command
"hyperopt-mongo-search --clear-existing bandit algo"
can only create a new experiment, and potentially deletes an existing one.

The command
"hyperopt-mongo-search --clear-existing --exp-key=EXPKEY bandit algo"
can only create a new experiment, and potentially deletes an existing one.


By default, MongoExperiment.run will try to save itself before returning. It
does so by pickling itself to a file called 'exp_key' in the fs collection.
Resuming means unpickling that file and calling run again.

The MongoExperiment instance itself is minimal (a key, a bandit, a bandit algo,
a workdir, a poll interval).  The only stateful element is the bandit algo.  The
difference between resume and start is in the handling of the bandit algo.


Worker
======

A worker looks up a job in a mongo database, maps that job document to a
runnable python object, calls that object, and writes the return value back to
the database.

A worker *reserves* a job by atomically identifying a document in the jobs
collection whose owner is None and whose state is 0, and setting the state to
1.  If it fails to identify such a job, it loops with a random sleep interval
of a few seconds and polls the database.

If hyperopt-mongo-worker is called with a --loop argument then it goes back to
the database after finishing a job to identify and perform another one.

CtrlObj
-------

The worker allocates a CtrlObj and passes it to bandit.evaluate in addition to
the subdocument found at job['spec'].  A bandit can use ctrl.info, ctrl.warn,
ctrl.error and so on like logger methods, and those messages will be written
to the mongo database (to job['logs']).  They are not written synchronously
though, they are written when the bandit.evaluate function calls
ctrl.checkpoint().

Ctrl.checkpoint does several things:
* flushes logging messages to the database
* updates the refresh_time
* optionally updates the result subdocument

The main_worker routine calls Ctrl.checkpoint(rval) once after the
bandit.evalute function has returned before setting the state to 2 or 3 to
finalize the job in the database.

"""

__authors__ = ["James Bergstra", "Dan Yamins"]
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import copy
import cPickle
import datetime
import hashlib
import logging
import optparse
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
from bson import SON


logger = logging.getLogger(__name__)

from .base import JOB_STATES
from .base import (JOB_STATE_NEW, JOB_STATE_RUNNING, JOB_STATE_DONE,
        JOB_STATE_ERROR)
from .base import Experiment
from .base import Trials
from .base import trials_from_docs
from .base import InvalidTrial
from .base import Ctrl
from .base import SONify
from .utils import json_call
import plotting


class OperationFailure(Exception):
    """Proxy that could be factored out if we also want to use CouchDB and
    JobmanDB classes with this interface
    """


class Shutdown(Exception):
    """
    Exception for telling mongo_worker loop to quit
    """


class InvalidMongoTrial(InvalidTrial):
    pass


class BanditSwapError(Exception):
    """Raised when the search program tries to change the bandit attached to
    an experiment.
    """


class ReserveTimeout(Exception):
    """No job was reserved in the alotted time
    """


def read_pw():
    username = 'hyperopt'
    password = open(os.path.join(os.getenv('HOME'), ".hyperopt")).read()[:-1]
    return dict(
            username=username,
            password=password)


def authenticate_for_db(db):
    d = read_pw()
    db.authenticate(d['username'], d['password'])


def parse_url(url, pwfile=None):
    """Unpacks a url of the form
        protocol://[username[:pw]]@hostname[:port]/db/collection

    :rtype: tuple of strings
    :returns: protocol, username, password, hostname, port, dbname, collection

    :note:
    If the password is not given in the url but the username is, then
    this function will read the password from file by calling
    ``open(pwfile).read()[:-1]``

    """

    protocol=url[:url.find(':')]
    ftp_url='ftp'+url[url.find(':'):]

    # -- parse the string as if it were an ftp address
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

    return (protocol, tmp.username, password, tmp.hostname, tmp.port, dbname,
            collection)


def connection_with_tunnel(host='localhost',
            auth_dbname='admin', port=27017,
            ssh=False, user='hyperopt', pw=None):
        if ssh:
            local_port=numpy.random.randint(low=27500, high=28000)
            # -- forward from local to remote machine
            ssh_tunnel = subprocess.Popen(
                    ['ssh', '-NTf', '-L',
                        '%i:%s:%i'%(local_port, '127.0.0.1', port),
                        host],
                    #stdin=subprocess.PIPE,
                    #stdout=subprocess.PIPE,
                    #stderr=subprocess.PIPE,
                    )
            # -- give the subprocess time to set up
            time.sleep(.5)
            connection = pymongo.Connection('127.0.0.1', local_port,
                    document_class=SON)
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
    """
    # MongoDB stores only to the nearest millisecond
    # This is mentioned in a footnote here:
    # http://api.mongodb.org/python/1.9%2B/api/bson/son.html#dt
    """
    now = datetime.datetime.utcnow()
    microsec = (now.microsecond//10**3)*(10**3)
    return datetime.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second, microsec)


class MongoJobs(object):
    """
    # Interface to a Jobs database structured like this
    #
    # Collections:
    #
    # db.jobs - structured {config_name, 'cmd', 'owner', 'book_time',
    #                  'refresh_time', 'state', 'exp_key', 'owner', 'result'}
    #    This is the collection that the worker nodes write to
    #
    # db.gfs - file storage via gridFS for all collections
    #
    """
    def __init__(self, db, jobs, gfs, conn, tunnel, config_name):
        self.db = db
        self.jobs = jobs
        self.gfs = gfs
        self.conn=conn
        self.tunnel=tunnel
        self.config_name = config_name

    # TODO: rename jobs -> coll throughout
    coll = property(lambda s : s.jobs)

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
    def new_from_connection_str(cls, conn_str, gfs_coll='fs', config_name='spec'):
        connection, tunnel, db, coll = connection_from_string(conn_str)
        gfs = gridfs.GridFS(db, collection=gfs_coll)
        return cls(db, coll, gfs, connection, tunnel, config_name)

    def __iter__(self):
        return self.jobs.find()

    def __len__(self):
        try:
            return self.jobs.count()
        except:
            return 0

    def create_jobs_indexes(self):
        jobs = self.db.jobs
        for k in ['exp_key', 'result.loss', 'book_time']:
            jobs.create_index(k)

    def create_drivers_indexes(self):
        drivers = self.db.drivers
        drivers.create_index('exp_key', unique=True)

    def create_indexes(self):
        self.create_jobs_indexes()
        self.create_drivers_indexes()

    def jobs_complete(self, cursor=False):
        c = self.jobs.find(spec=dict(state=JOB_STATE_DONE))
        return c if cursor else list(c)

    def jobs_error(self, cursor=False):
        c = self.jobs.find(spec=dict(state=JOB_STATE_ERROR))
        return c if cursor else list(c)

    def jobs_running(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(spec=dict(state=JOB_STATE_RUNNING)))
        #TODO: mark some as MIA
        rval = [r for r in rval if not r.get('MIA', False)]
        return rval

    def jobs_dead(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(spec=dict(state=JOB_STATE_RUNNING)))
        #TODO: mark some as MIA
        rval = [r for r in rval if r.get('MIA', False)]
        return rval

    def jobs_queued(self, cursor=False):
        c = self.jobs.find(spec=dict(state=JOB_STATE_NEW))
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
                for name, file_id in d.get('_attachments', []):
                    self.gfs.delete(file_id)
                logger.info('deleting job %s' % d['_id'])
                self.jobs.remove(d, safe=safe)
        except pymongo.errors.OperationFailure, e:
            raise OperationFailure(e)

    def delete_all_error_jobs(self, safe=True):
        return self.delete_all(cond={'state': JOB_STATE_ERROR}, safe=safe)

    def reserve(self, host_id, cond=None, exp_key=None):
        now = coarse_utcnow()
        if cond is None:
            cond = {}
        else:
            cond = copy.copy(cond) #copy is important, will be modified, but only the top-level

        if exp_key is not None:
            cond['exp_key'] = exp_key

        if 'owner' not in cond:
            cond['owner'] = None

        if cond['owner'] is not None:
            raise ValueError('refusing to reserve owned job')
        try:
            rval = self.jobs.find_and_modify(
                cond,
                {'$set':
                    {'owner': host_id,
                     'book_time': now,
                     'state': JOB_STATE_RUNNING,
                     'refresh_time': now,
                     }
                 },
                new=True,
                safe=True,
                upsert=False)
        except pymongo.errors.OperationFailure, e:
            logger.error('Error during reserve_job: %s'%str(e))
            rval = None
        return rval

    def refresh(self, doc, safe=False):
        self.update(doc, dict(refresh_time=coarse_utcnow()), safe=False)

    def update(self, doc, dct, safe=True, collection=None):
        """Return union of doc and dct, after making sure that dct has been
        added to doc in `collection`.

        This function does not modify either `doc` or `dct`.

        safe=True means error-checking is done. safe=False means this function will succeed
        regardless of what happens with the db.
        """
        if collection is None:
            collection = self.coll

        dct = copy.deepcopy(dct)
        if '_id' in dct:
            raise ValueError('cannot update the _id field')
        if 'version' in dct:
            raise ValueError('cannot update the version field')
        if '_id' not in doc:
            raise ValueError('doc must have an "_id" key to be updated')

        if 'version' in doc:
            doc_query = dict(_id=doc['_id'], version=doc['version'])
            dct['version'] = doc['version']+1
        else:
            doc_query = dict(_id=doc['_id'])
            dct['version'] = 1
        try:
            # warning - if doc matches nothing then this function succeeds
            # N.B. this matches *at most* one entry, and possibly zero
            collection.update(
                    doc_query,
                    {'$set': dct},
                    safe=True,
                    upsert=False,
                    multi=False,)
        except pymongo.errors.OperationFailure, e:
            # translate pymongo failure into generic failure
            raise OperationFailure(e)

        # update doc in-place to match what happened on the server side
        doc.update(dct)

        if safe:
            server_doc = collection.find_one(
                    dict(_id=doc['_id'], version=doc['version']))
            if server_doc is None:
                raise OperationFailure('updated doc not found : %s'
                        % str(doc))
            elif server_doc != doc:
                if 0:# This is all commented out because it is tripping on the fact that
                    # str('a') != unicode('a').
                    # TODO: eliminate false alarms and catch real ones
                    mismatching_keys = []
                    for k, v in server_doc.items():
                        if k in doc:
                            if doc[k] != v:
                                mismatching_keys.append((k, v, doc[k]))
                        else:
                            mismatching_keys.append((k, v, '<missing>'))
                    for k,v in doc.items():
                        if k not in server_doc:
                            mismatching_keys.append((k, '<missing>', v))

                    raise OperationFailure('local and server doc documents are out of sync: %s'%
                            repr((doc, server_doc, mismatching_keys)))
        return doc

    def attachment_names(self, doc):
        return [a[0] for a in doc.get('_attachments', [])]

    def set_attachment(self, doc, blob, name, collection=None):
        """Attach potentially large data string `blob` to `doc` by name `name`

        blob must be a string

        doc must have been saved in some collection (must have an _id), but not
        necessarily the jobs collection.

        name must be a string

        Returns None
        """

        # If there is already a file with the given name for this doc, then we will delete it
        # after writing the new file
        attachments = doc.get('_attachments', [])
        name_matches = [a for a in attachments if a[0] == name]

        # the filename is set to something so that fs.list() will display the file
        new_file_id = self.gfs.put(blob, filename='%s_%s' % (doc['_id'], name))
        logger.info('stored blob of %i bytes with id=%s and filename %s_%s' % (
            len(blob), str(new_file_id), doc['_id'], name))

        new_attachments = ([a for a in attachments if a[0] != name]
                + [(name, new_file_id)])

        try:
            ii = 0
            doc = self.update(doc, {'_attachments': new_attachments},
                    collection=collection)
            # there is a database leak until we actually delete the files that
            # are no longer pointed to by new_attachments
            while ii < len(name_matches):
                self.gfs.delete(name_matches[ii][1])
                ii += 1
        except:
            while ii < len(name_matches):
                logger.warning("Leak during set_attachment: old_file_id=%s" % (
                    name_matches[ii][1]))
                ii += 1
            raise
        assert len([n for n in self.attachment_names(doc) if n == name]) == 1
        #return new_file_id

    def get_attachment(self, doc, name):
        """Retrieve data attached to `doc` by `attach_blob`.

        Raises OperationFailure if `name` does not correspond to an attached blob.

        Returns the blob as a string.
        """
        attachments = doc.get('_attachments', [])
        file_ids = [a[1] for a in attachments if a[0] == name]
        if not file_ids:
            raise OperationFailure('Attachment not found: %s' % name)
        if len(file_ids) > 1:
            raise OperationFailure('multiple name matches', (name, file_ids))
        return self.gfs.get(file_ids[0]).read()

    def delete_attachment(self, doc, name, collection=None):
        attachments = doc.get('_attachments', [])
        file_id = None
        for i,a in enumerate(attachments):
            if a[0] == name:
                file_id = a[1]
                break
        if file_id is None:
            raise OperationFailure('Attachment not found: %s' % name)
        #print "Deleting", file_id
        del attachments[i]
        self.update(doc, {'_attachments':attachments}, collection=collection)
        self.gfs.delete(file_id)


class MongoTrials(Trials):
    """Trials maps on to an entire mongo collection. It's basically a wrapper
    around MongoJobs for now.

    As a concession to performance, this object permits trial filtering based
    on the exp_key, but I feel that's a hack. The case of `cmd` is similar--
    the exp_key and cmd are semantically coupled.
    """
    async = True

    def __init__(self, arg, exp_key=None, cmd=None, workdir=None,
            refresh=True):
        if isinstance(arg, MongoJobs):
            self.handle = arg
        else:
            connection_string = arg
            self.handle = MongoJobs.new_from_connection_str(connection_string)
        self.handle.create_indexes()
        self._exp_key = exp_key
        self.cmd = cmd
        self.workdir = workdir
        if refresh:
            self.refresh()

    def view(self, exp_key=None, cmd=None, workdir=None, refresh=True):
        rval = self.__class__(self.handle,
                exp_key=self._exp_key if exp_key is None else exp_key,
                cmd=self.cmd if cmd is None else cmd,
                workdir=self.workdir if workdir is None else workdir,
                refresh=refresh)
        return rval

    def refresh(self):
        exp_key = self._exp_key
        if exp_key != None:
            query = {'exp_key' : exp_key}
        else:
            query = {}
        t0 = time.time()
        all_jobs = list(self.handle.jobs.find(query))
        logger.info('Refresh took %f seconds' % (time.time() - t0))
        id_jobs = [(j['_id'], j)
            for j in all_jobs
            if j['state'] != JOB_STATE_ERROR]
        logger.info('skipping %i error jobs' % (len(all_jobs) - len(id_jobs)))
        jarray = numpy.array([j['_id'] for j in all_jobs])
        jobsort = jarray.argsort()
        id_jobs = [id_jobs[idx] for idx in jobsort]
        self._trials = [j for (_id, j) in id_jobs]
        self._specs = [j['spec'] for (_id, j) in id_jobs]
        self._results = [j['result'] for (_id, j) in id_jobs]
        self._miscs = [j['misc'] for (_id, j) in id_jobs]

    def _insert_trial_docs(self, docs):
        rval = []
        for doc in docs:
            rval.append(self.handle.jobs.insert(doc, safe=True))
        return rval

    def count_by_state_unsynced(self, arg):
        exp_key = self._exp_key
        # TODO: consider searching by SON rather than dict
        if isinstance(arg, int):
            if arg not in JOB_STATES:
                raise ValueError('invalid state', arg)
            query = dict(state=arg)
        else:
            assert hasattr(arg, '__iter__')
            states = list(arg)
            assert all([x in JOB_STATES for x in states])
            query = dict(state={'$in': states})
        if exp_key != None:
            query['exp_key'] = exp_key
        rval = self.handle.jobs.find(query).count()
        return rval

    def delete_all(self):
        if self._exp_key:
            cond = cond={'exp_key': self._exp_key}
        else:
            cond = {}
        # -- remove all documents matching condition
        self.handle.delete_all(cond)
        gfs = self.handle.gfs
        for filename in gfs.list():
            gfs.delete(gfs.get_last_version(filename)._id)
        self.refresh()

    def new_trial_ids(self, N):
        db = self.handle.db
        if self._exp_key is None:
            # -- docs say you can't upsert an empty document
            query = {'a': 0}
        else:
            query = {'exp_key':self._exp_key}
        doc = db.job_ids.find_and_modify(
                query,
                {'$inc' : {'last_id': N}},
                upsert=True,
                safe=True)
        lid = doc.get('last_id', 0)
        return range(lid, lid + N)

    @property
    def attachments(self):
        """
        Support syntax for load:  self.attachments[name]
        Support syntax for store: self.attachments[name] = value
        """
        gfs = self.handle.gfs
        class Attachments(object):
            def __contains__(_self, name):
                return gfs.exists(filename=name)

            def __getitem__(_self, name):
                try:
                    rval = gfs.get_version(name).read()
                    return rval
                except gridfs.NoFile, e:
                    raise KeyError(name)

            def __setitem__(_self, name, value):
                if gfs.exists(filename=name):
                    gout = gfs.get_last_version(name)
                    gfs.delete(gout._id)
                gfs.put(value, filename=name)

            def __delitem__(_self, name):
                gout = gfs.get_last_version(name)
                gfs.delete(gout._id)

        return Attachments()


class MongoWorker(object):
    poll_interval = 3.0  # -- seconds
    workdir = None

    def __init__(self, mj,
            poll_interval=poll_interval,
            workdir=workdir,
            exp_key=None):
        """
        mj - MongoJobs interface to jobs collection
        poll_interval - seconds
        workdir - string
        exp_key - restrict reservations to this key
        """
        self.mj = mj
        self.poll_interval = poll_interval
        self.workdir = workdir
        self.exp_key = exp_key

    def run_one(self, host_id=None, reserve_timeout=None):
        if host_id == None:
            host_id = '%s:%i'%(socket.gethostname(), os.getpid()),
        job = None
        start_time = time.time()
        mj = self.mj
        while job is None:
            if (reserve_timeout
                    and (time.time() - start_time) > reserve_timeout):
                raise ReserveTimeout()
            job = mj.reserve(host_id, exp_key=self.exp_key)
            if not job:
                interval = (1 +
                        numpy.random.rand()
                        * (float(self.poll_interval) - 1.0))
                logger.info('no job found, sleeping for %.1fs' % interval)
                time.sleep(interval)

        logger.info('job found: %s' % str(job))

        # -- don't let the cmd mess up our trial object
        spec = copy.deepcopy(job['spec'])

        ctrl = MongoCtrl(
                # XXX: should the job's exp_key be used here?
                trials=MongoTrials(mj, exp_key=self.exp_key, refresh=False),
                read_only=False,
                current_trial=job)
        if self.workdir is None:
            workdir = job['misc'].get('workdir', os.getcwd())
            if workdir is None:
                workdir = ''
            workdir = os.path.join(workdir, str(job['_id']))
        else:
            workdir = self.workdir
        workdir = os.path.expanduser(workdir)
        if not os.path.isdir(workdir):
            os.makedirs(workdir)
        os.chdir(workdir)
        cmd = job['misc']['cmd']
        cmd_protocol = cmd[0]
        try:
            if cmd_protocol == 'cpickled fn':
                worker_fn = cPickle.loads(cmd[1])
            elif cmd_protocol == 'call evaluate':
                bandit = cPickle.loads(cmd[1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'token_load':
                cmd_toks = cmd[1].split('.')
                cmd_module = '.'.join(cmd_toks[:-1])
                worker_fn = exec_import(cmd_module, cmd[1])
            elif cmd_protocol == 'bandit_json evaluate':
                bandit = json_call(cmd[1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'driver_attachment':
                #name = 'driver_attachment_%s' % job['exp_key']
                blob = ctrl.trials.attachments[cmd[1]]
                bandit_name, bandit_args, bandit_kwargs = cPickle.loads(blob)
                worker_fn = json_call(bandit_name,
                        args=bandit_args,
                        kwargs=bandit_kwargs).evaluate
            else:
                raise ValueError('Unrecognized cmd protocol', cmd_protocol)

            result = SONify(worker_fn(spec, ctrl))
        except Exception, e:
            #XXX: save exception to database, but if this fails, then
            #      at least raise the original traceback properly
            logger.info('job exception: %s' % str(e))
            ctrl.checkpoint()
            mj.update(job,
                    {'state': JOB_STATE_ERROR,
                    'error': (str(type(e)), str(e))},
                    safe=True)
            raise

        logger.info('job finished: %s' % str(job['_id']))
        ctrl.checkpoint(result)
        mj.update(job, {'state': JOB_STATE_DONE}, safe=True)


class MongoCtrl(Ctrl):
    """
    Attributes:

    current_trial - current job document
    jobs - MongoJobs object in which current_trial resides
    read_only - True means don't change the db

    """
    def __init__(self, trials, current_trial, read_only):
        self.trials = trials
        self.current_trial = current_trial
        self.read_only = read_only

    def debug(self, *args, **kwargs):
        # XXX: This is supposed to log to db
        return logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        # XXX: This is supposed to log to db
        return logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        # XXX: This is supposed to log to db
        return logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        # XXX: This is supposed to log to db
        return logger.error(*args, **kwargs)

    def checkpoint(self, result=None):
        if not self.read_only:
            handle = self.trials.handle
            handle.refresh(self.current_trial)
            if result is not None:
                return handle.update(self.current_trial, dict(result=result))

    @property
    def attachments(self):
        """
        Support syntax for load:  self.attachments[name]
        Support syntax for store: self.attachments[name] = value
        """
        handle = self.trials.handle
        class Attachments(object):
            def __contains__(_self, name):
                names = handle.attachment_names(
                        doc=self.current_trial)
                return name in names

            def __getitem__(_self, name):
                try:
                    return handle.get_attachment(
                        doc=self.current_trial,
                        name=name)
                except OperationFailure:
                    raise KeyError(name)

            def __setitem__(_self, name, value):
                handle.set_attachment(
                    doc=self.current_trial,
                    blob=value,
                    name=name,
                    collection=handle.db.jobs)

        return Attachments()

    @property
    def set_attachment(self):
        # XXX: Is there a better deprecation error?
        raise RuntimeError(
            'set_attachment deprecated. Use `self.attachments[name] = value`')


def exec_import(cmd_module, cmd):
    worker_fn = None
    exec('import %s; worker_fn = %s' % (cmd_module, cmd))
    return worker_fn


def as_mongo_str(s):
    if s.startswith('mongo://'):
        return s
    else:
        return 'mongo://%s' % s


def main_worker_helper(options, args):
    N = int(options.max_jobs)

    if N > 1:
        def sighandler_shutdown(signum, frame):
            logger.info('Caught signal %i, shutting down.' % signum)
            raise Shutdown(signum)
        signal.signal(signal.SIGINT, sighandler_shutdown)
        signal.signal(signal.SIGHUP, sighandler_shutdown)
        signal.signal(signal.SIGTERM, sighandler_shutdown)
        proc = None
        cons_errs = 0
        while N and cons_errs < int(options.max_consecutive_failures):
            try:
                # recursive Popen, dropping N from the argv
                # By using another process to run this job
                # we protect ourselves from memory leaks, bad cleanup
                # and other annoying details.
                # The tradeoff is that a large dataset must be reloaded once for
                # each subprocess.
                sub_argv = [sys.argv[0],
                        '--poll-interval=%s' % options.poll_interval,
                        '--max-jobs=1',
                        '--mongo=%s' % options.mongo]
                if options.workdir is not None:
                    sub_argv.append('--workdir=%s' % options.workdir)
                if options.exp_key is not None:
                    sub_argv.append('--exp-key=%s' % options.exp_key)
                proc = subprocess.Popen(sub_argv)
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
                cons_errs += 1
            else:
                cons_errs = 0
            N -= 1
        logger.info("exiting with N=%i after %i consecutive exceptions" %(
            N, cons_errs))
    elif N == 1:
        # XXX: the name of the jobs collection is a parameter elsewhere,
        #      so '/jobs' should not be hard-coded here
        mj = MongoJobs.new_from_connection_str(
                as_mongo_str(options.mongo) + '/jobs')

        mworker = MongoWorker(mj,
                float(options.poll_interval),
                workdir=options.workdir,
                exp_key=options.exp_key)
        mworker.run_one(reserve_timeout=float(options.reserve_timeout))
    else:
        parser.print_help()
        return -1


def main_worker():
    parser = optparse.OptionParser(usage="%prog [options]")

    parser.add_option("--max-consecutive-failures",
            dest="max_consecutive_failures",
            metavar='N',
            default=4,
            help="stop if N consecutive jobs fail (default: 4)",
            )
    parser.add_option("--exp-key",
            dest='exp_key',
            default = None,
            metavar='str',
            help="identifier for this workers's jobs")
    parser.add_option("--poll-interval",
            dest='poll_interval',
            metavar='N',
            default=5,
            help="check work queue every 1 < T < N seconds (default: 5")
    parser.add_option("--max-jobs",
            dest='max_jobs',
            default=sys.maxint,
            help="stop after running this many jobs (default: inf)")
    parser.add_option("--mongo",
            dest='mongo',
            default='localhost/hyperopt',
            help="<host>[:port]/<db> for IPC and job storage")
    parser.add_option("--reserve-timeout",
            dest='reserve_timeout',
            metavar='T',
            default=120.0,
            help="poll database for up to T seconds to reserve a job")
    parser.add_option("--workdir",
            dest="workdir",
            default=None,
            help="root workdir (default: load from mongo)",
            metavar="DIR")

    (options, args) = parser.parse_args()

    if args:
        parser.print_help()
        return -1

    return main_worker_helper(options, args)


def bandit_from_options(options):
    #
    # Construct bandit
    #
    bandit_name = options.bandit
    if options.bandit_argfile:
        bandit_argfile_text = open(options.bandit_argfile).read()
        bandit_argv, bandit_kwargs = cPickle.loads(bandit_argfile_text)
    else:
        bandit_argfile_text = ''
        bandit_argv, bandit_kwargs = (), {}
    bandit = json_call(bandit_name, bandit_argv, bandit_kwargs)
    return (bandit,
            (bandit_name, bandit_argv, bandit_kwargs),
            bandit_argfile_text)



def algo_from_options(options, bandit):
    #
    # Construct algo
    #
    algo_name = options.bandit_algo
    if options.bandit_algo_argfile:
        # in theory this is easy just as above.
        # need tests though, and it's just not done yet.
        raise NotImplementedError('Option: --bandit-algo-argfile')
    else:
        algo_argfile_text = ''
        algo_argv, algo_kwargs = (), {}
    algo = json_call(algo_name, (bandit,) + algo_argv, algo_kwargs)
    return (algo,
            (algo_name, (bandit,) + algo_argv, algo_kwargs),
            algo_argfile_text)


def expkey_from_options(options, bandit_stuff, algo_stuff):
    #
    # Determine exp_key
    #
    if None is options.exp_key:
        # -- argfile texts
        bandit_name = bandit_stuff[1][0]
        algo_name = algo_stuff[1][0]
        bandit_argfile_text = bandit_stuff[2]
        algo_argfile_text = algo_stuff[2]
        if bandit_argfile_text or algo_argfile_text:
            m = hashlib.md5()
            m.update(bandit_argfile_text)
            m.update(algo_argfile_text)
            exp_key = '%s/%s[arghash:%s]' % (
                    bandit_name, algo_name, m.hexdigest())
            del m
        else:
            exp_key = '%s/%s' % (bandit_name, algo_name)
    else:
        exp_key = options.exp_key
    return exp_key


def main_search_helper(options, args, input=input, cmd_type=None):
    """
    input is an argument so that unittest can replace stdin
    """
    options.bandit = args[0]
    options.bandit_algo = args[1]

    bandit_stuff = bandit_from_options(options)
    bandit, bandit_NAK, bandit_argfile_text = bandit_stuff
    bandit_name, bandit_args, bandit_kwargs = bandit_NAK

    algo_stuff = algo_from_options(options, bandit)
    algo, algo_NAK, algo_argfile_text = algo_stuff
    algo_name, algo_args, algo_kwargs = algo_NAK

    exp_key = expkey_from_options(options, bandit_stuff, algo_stuff)

    trials = MongoTrials(as_mongo_str(options.mongo) + '/jobs', exp_key)

    if options.clear_existing:
        print >> sys.stdout, "Are you sure you want to delete",
        print >> sys.stdout, ("all %i jobs with exp_key: '%s' ?"
                % (
                    trials.handle.db.jobs.find({'exp_key':exp_key}).count(),
                    str(exp_key)))
        print >> sys.stdout, '(y/n)'
        y, n = 'y', 'n'
        if input() != 'y':
            print >> sys.stdout, "aborting"
            del self
            return 1
        trials.delete_all()

    #
    # Construct MongoExperiment
    #
    if bandit_argfile_text or algo_argfile_text or cmd_type=='D.A.':
        aname = 'driver_attachment_%s.pkl' % exp_key
        worker_cmd = ('driver_attachment', aname)
        if aname in trials.attachments:
            atup = cPickle.loads(trials.attachments[aname])
            if bandit_NAK != atup:
                raise BanditSwapError((bandit_NAK, atup))
        else:
            blob = cPickle.dumps(bandit_NAK)
            trials.attachments[aname] = blob
    else:
        worker_cmd = ('bandit_json evaluate', bandit_name)

    algo.cmd = worker_cmd
    algo.workdir=options.workdir

    self = Experiment(trials,
        bandit_algo=algo,
        poll_interval_secs=(int(options.poll_interval))
            if options.poll_interval else 5,
        max_queue_len=options.max_queue_len)

    self.run(options.steps, block_until_done=options.block)


def main_search():
    parser = optparse.OptionParser(
            usage="%prog [options] [<bandit> <bandit_algo>]")
    parser.add_option("--clear-existing",
            action="store_true",
            dest="clear_existing",
            default=False,
            help="clear all jobs with the given exp_key")
    parser.add_option("--exp-key",
            dest='exp_key',
            default = None,
            metavar='str',
            help="identifier for this driver's jobs")
    parser.add_option('--force-lock',
            action="store_true",
            dest="force_lock",
            default=False,
            help="ignore concurrent experiments using same exp_key (only do this after a crash)")
    parser.add_option("--mongo",
            dest='mongo',
            default='localhost/hyperopt',
            help="<host>[:port]/<db> for IPC and job storage")
    parser.add_option("--poll-interval",
            dest='poll_interval',
            metavar='N',
            default=None,
            help="check work queue every N seconds (default: 5")
    parser.add_option("--no-save-on-exit",
            action="store_false",
            dest="save_on_exit",
            default=True,
            help="save driver state to mongo on exit")
    parser.add_option("--steps",
            dest='steps',
            default=sys.maxint,
            help="exit after queuing this many jobs (default: inf)")
    parser.add_option("--workdir",
            dest="workdir",
            default=os.path.expanduser('~/.hyperopt.workdir'),
            help="direct hyperopt-mongo-worker to chdir here",
            metavar="DIR")
    parser.add_option("--block",
            dest="block",
            action="store_true",
            default=False,
            help="block return until all queue is empty")
    parser.add_option("--bandit-argfile",
            dest="bandit_argfile",
            default=None,
            help="path to file containing arguments bandit constructor\n"
                 "file format: pickle of dictionary containing two keys,\n"
                 "  {'args' : tuple of positional arguments,\n"
                 "   'kwargs' : dictionary of keyword arguments}")
    parser.add_option("--bandit-algo-argfile",
            dest="bandit_algo_argfile",
            default=None,
            help="path to file containing arguments for bandit_algo "
                  "constructor.  File format is pickled dictionary containing "
                  "two keys:\n"
                  "  'args', a tuple of positional arguments, and \n"
                  "  'kwargs', a dictionary of keyword arguments. \n"
                  "NOTE: instantiated bandit is pre-pended as first element"
                  " of arg tuple.")
    parser.add_option("--max-queue-len",
            dest="max_queue_len",
            default=1,
            help="maximum number of jobs to allow in queue")

    (options, args) = parser.parse_args()

    if len(args) > 2:
        parser.print_help()
        return -1

    return main_search_helper(options, args)


def main_show_helper(options, args):
    if options.trials_pkl:
        trials = cPickle.load(open(options.trials_pkl))
    else:
        bandit_stuff = bandit_from_options(options)
        bandit, (bandit_name, bandit_args, bandit_kwargs), bandit_algo_argfile\
                = bandit_stuff

        algo_stuff = algo_from_options(options, bandit)
        algo, (algo_name, algo_args, algo_kwargs), algo_algo_argfile\
                = algo_stuff

        exp_key = expkey_from_options(options, bandit_stuff, algo_stuff)

        trials = MongoTrials(as_mongo_str(options.mongo) + '/jobs', exp_key)

    cmd = args[0]
    if 'history' == cmd:
        if 0:
            import matplotlib.pyplot as plt
            self.refresh_trials_results()
            yvals, colors = zip(*[(1 - r.get('best_epoch_test', .5), 'g')
                for y, r in zip(self.losses(), self.results) if y is not None])
            plt.scatter(range(len(yvals)), yvals, c=colors)
        return plotting.main_plot_history(trials)
    elif 'histogram' == cmd:
        return plotting.main_plot_histogram(trials)
    elif 'dump' == cmd:
        raise NotImplementedError('TODO: dump jobs db to stdout as JSON')
    elif 'dump_pickle' == cmd:
        cPickle.dump(trials_from_docs(trials.trials),
                open(args[1], 'w'))
    elif 'vars' == cmd:
        return plotting.main_plot_vars(trials)
    else:
        logger.error("Invalid cmd %s" % cmd)
        parser.print_help()
        print """Current supported commands are history, histogram, vars
        """
        return -1


def main_show():
    parser = optparse.OptionParser(
            usage="%prog [options] cmd [...]")
    parser.add_option("--exp-key",
            dest='exp_key',
            default = None,
            metavar='str',
            help="identifier for this driver's jobs")
    parser.add_option("--bandit",
            dest='bandit',
            default = None,
            metavar='json',
            help="identifier for the bandit solved by the experiment")
    parser.add_option("--bandit-argfile",
            dest="bandit_argfile",
            default=None,
            help="path to file containing arguments bandit constructor\n"
                 "file format: pickle of dictionary containing two keys,\n"
                 "  {'args' : tuple of positional arguments,\n"
                 "   'kwargs' : dictionary of keyword arguments}")
    parser.add_option("--bandit-algo",
            dest='bandit_algo',
            default = None,
            metavar='json',
            help="identifier for the optimization algorithm for experiment")
    parser.add_option("--bandit-algo-argfile",
            dest="bandit_algo_argfile",
            default=None,
            help="path to file containing arguments for bandit_algo "
                  "constructor.  File format is pickled dictionary containing "
                  "two keys:\n"
                  "  'args', a tuple of positional arguments, and \n"
                  "  'kwargs', a dictionary of keyword arguments. \n"
                  "NOTE: instantiated bandit is pre-pended as first element"
                  " of arg tuple.")
    parser.add_option("--mongo",
            dest='mongo',
            default='localhost/hyperopt',
            help="<host>[:port]/<db> for IPC and job storage")
    parser.add_option("--trials",
            dest="trials_pkl",
            default="",
            help="local trials file (e.g. created by dump_pickle command)")
    parser.add_option("--workdir",
            dest="workdir",
            default=os.path.expanduser('~/.hyperopt.workdir'),
            help="check for worker files here",
            metavar="DIR")

    (options, args) = parser.parse_args()

    try:
        cmd = args[0]
    except:
        parser.print_help()
        return -1

    return main_show_helper(options, args)

