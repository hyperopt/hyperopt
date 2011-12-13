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

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import copy
import cPickle
import datetime
import hashlib
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
from bson import BSON, SON

import base
import utils

logger = logging.getLogger(__name__)

STATE_NEW = 0
STATE_RUNNING = 1
STATE_DONE = 2
STATE_ERROR = 3
JOB_STATES = [STATE_NEW,
              STATE_RUNNING,
              STATE_DONE,
              STATE_ERROR]


class OperationFailure(Exception):
    """Proxy that could be factored out if we also want to use CouchDB and
    JobmanDB classes with this interface
    """


class Shutdown(Exception):
    """
    Exception for telling mongo_worker loop to quit
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
    """
    # MongoDB stores only to the nearest millisecond
    # This is mentioned in a footnote here:
    # http://api.mongodb.org/python/1.9%2B/api/bson/son.html#dt
    """
    now = datetime.datetime.utcnow()
    microsec = (now.microsecond//10**3)*(10**3)
    return datetime.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second, microsec)


def create_jobs_indexes(jobs):
    for k in ['exp_key', 'result.loss', 'book_time']:
        jobs.create_index(k)


def create_drivers_indexes(drivers):
    for k in ['exp_key']:
        drivers.create_index(k)

 
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

    def jobs_complete(self, cursor=False):
        c = self.jobs.find(spec=dict(state=STATE_DONE))
        return c if cursor else list(c)
    def jobs_error(self, cursor=False):
        c = self.jobs.find(spec=dict(state=STATE_ERROR))
        return c if cursor else list(c)
    def jobs_running(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(spec=dict(state=STATE_RUNNING)))
        #TODO: mark some as MIA
        rval = [r for r in rval if not r.get('MIA', False)]
        return rval
    def jobs_dead(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(spec=dict(state=STATE_RUNNING)))
        #TODO: mark some as MIA
        rval = [r for r in rval if r.get('MIA', False)]
        return rval
    def jobs_queued(self, cursor=False):
        c = self.jobs.find(spec=dict(state=STATE_NEW))
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
        return self.delete_all(cond={'state': STATE_ERROR}, safe=safe)

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
                    {'owner': host_id,
                     'book_time': now,
                     'state': STATE_RUNNING,
                     'refresh_time': now,
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


class MongoExperiment(base.Experiment):
    """
    This experiment uses a Mongo collection to store
    - self.trials
    - self.results

    """
    def __init__(self, bandit_algo, mongo_handle, workdir, exp_key, cmd,
            poll_interval_secs=10,
            save_interval_secs=3.0,
            max_queue_len=1):

        if isinstance(mongo_handle, str):
            self.mongo_handle = MongoJobs.new_from_connection_str(
                    mongo_handle,
                    config_name='spec')
        else:
            #N.B. self.mongo_handle may be None.
            #     This is the case if we're unpickling.
            self.mongo_handle = mongo_handle

        base.Experiment.__init__(self, bandit_algo)
        self.workdir = workdir               # can be changed
        self.poll_interval_secs = poll_interval_secs  # can be changed
        self.save_interval_secs = save_interval_secs
        self.max_queue_len = max_queue_len   # can be changed
        self.exp_key = exp_key               # don't change this
        self.cmd = cmd                       # don't change this
        self._locks = 0                      # this is managed internally

    def __getstate__(self):
        rval = dict(self.__dict__)
        del rval['mongo_handle']
        return rval

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        if 'trials' not in dct:
            assert 'results' not in dct
            self.trials = []
            self.results = []

    def refresh_trials_results(self):
        query = {'exp_key': self.exp_key}
        all_jobs = list(self.mongo_handle.jobs.find(query))
        logger.info('skipping %i error jobs'
                % len([j for j in all_jobs if j['state'] == STATE_ERROR]))
        id_jobs = [(j['_id'], j)
            for j in all_jobs if j['state'] != STATE_ERROR]
        id_jobs.sort()
        self.trials[:] = [j['spec'] for (_id, j) in id_jobs]
        self.results[:] = [j['result'] for (_id, j) in id_jobs]

    def queue_extend(self, trial_configs, skip_dups=True):
        if skip_dups:
            new_configs = []
            for config in trial_configs:
                #XXX: This will basically never work
                #     now that TheanoBanditAlgo puts a _config_id into
                #     each suggestion
                query = self.mongo_handle.jobs.find(dict(spec=config))
                if query.count():
                    matches = list(query)
                    assert len(matches) == 1
                    logger.info('Skipping duplicate trial')
                else:
                    new_configs.append(config)
            trial_configs = new_configs
        rval = []
        # this tells the mongo-worker how to evaluate the job
        for config in trial_configs:
            to_insert = dict(
                    state=STATE_NEW,
                    exp_key=self.exp_key,
                    cmd=self.cmd,
                    owner=None,
                    spec=config,
                    result=self.bandit_algo.bandit.new_result(),
                    version=0,
                    )
            rval.append(self.mongo_handle.jobs.insert(to_insert, safe=True))
        return rval

    def queue_len(self, states=STATE_NEW):
        # TODO: consider searching by SON rather than dict
        if isinstance(states,int):
            assert states in JOB_STATES
            query = dict(state=states, exp_key=self.exp_key)
        else:
            assert hasattr(states, '__iter__')
            states = list(states)
            assert all([x in JOB_STATES for x in states])
            query = dict(state={'$in': states},
                         exp_key=self.exp_key)
        rval = self.mongo_handle.jobs.find(query).count()
        logger.debug('Queue len: %i' % rval)
        return rval

    def run(self, N, block_until_done=False):
        algo = self.bandit_algo
        n_queued = 0
        while n_queued < N:
            while self.queue_len() < self.max_queue_len:
                self.refresh_trials_results()
                t0 = time.time()
                # just ask for one job at a time
                suggestions = algo.suggest(self.trials, self.results, 1)
                t1 = time.time()
                logger.info('algo suggested trial: %s (in %.2f seconds)'
                        % (str(suggestions[0]), t1 - t0))
                try:
                    new_suggestions = self.queue_extend(suggestions)
                except:
                    logger.error('Problem with suggestion: %s' % (
                        suggestions))
                    raise
                n_queued += len(new_suggestions)
            time.sleep(self.poll_interval_secs)

        if block_until_done:
            while self.queue_len(states=[STATE_NEW, STATE_RUNNING]) > 0:
                msg = 'Waiting for %d jobs to finish ...' % self.queue_len()
                logger.info(msg)
                time.sleep(self.poll_interval_secs)
            logger.info('Queue empty, exiting run.')
        else:
            msg = 'Exiting run, not waiting for %d jobs.' % self.queue_len()
            logger.info(msg)

        self.refresh_trials_results()

    def clear_from_db(self):
        if not self._locks:
            raise OperationFailure('need lock to clear db')

        ddoc = self.ddoc_get()
        # remove attached states, bandits, etc.
        for name in self.mongo_handle.attachment_names(ddoc):
            self.mongo_handle.delete_attachment(ddoc, name,
                    collection=self.mongo_handle.db.drivers)

        # delete any jobs from a previous driver
        self.mongo_handle.delete_all(cond={'exp_key': self.exp_key})

        # remove the ddoc itself
        self.mongo_handle.db.drivers.remove(ddoc, safe=True)

        # we deleted the document used for locking, so we no longer have a
        # lock
        self._locks = 0


    def owner_label(self):
        # Don't make this an attribute, because after unpickling it has to be
        # updated.
        return '%s:%i:%i' % (socket.gethostname(), os.getpid(), id(self))

    def ddoc_get(self):
        """Get the document from the drivers collection for this experiment.

        Returns None if no such document exists.
        """
        query = self.mongo_handle.db.drivers.find(dict(exp_key=self.exp_key))
        assert query.count() < 2, query.count()
        if query.count() == 0:
            self.ddoc_init()
        return self.mongo_handle.db.drivers.find_one(
                {'exp_key': self.exp_key})

    def ddoc_init(self):
        """
        Initializes the document in the drivers collection for this
        experiment.
        """
        query = self.mongo_handle.db.drivers.find(dict(exp_key=self.exp_key))
        assert query.count() == 0, query.count()
        logger.info('inserting config document')
        config = dict(
                exp_key=self.exp_key,
                owner=None,
                workdir=self.workdir)
        self.mongo_handle.db.drivers.insert(config)

    def ddoc_lock(self, force=False):
        """Acquire a lock on the drivers document
        """
        if self._locks == 0:
            try:
                self.mongo_handle.db.drivers.update(
                        dict(exp_key=self.exp_key,
                            owner=None),
                        {'$set':
                            { 'owner': self.owner_label() } },
                        safe=True, upsert=False, multi=False)
            except pymongo.errors.OperationFailure, e:
                raise OperationFailure('failed to obtain lock (case 1)')
            ddoc = self.mongo_handle.db.drivers.find_one(
                    {'exp_key': self.exp_key, 'owner': self.owner_label()})
            if ddoc is None:
                raise OperationFailure('failed to obtain lock (case 2)')
            self._locks = 1
        else:
            self._locks += 1

    def ddoc_release(self):
        """Release the lock """
        if self._locks > 1:
            self._locks -= 1
        elif self._locks == 1:
            ddoc = self.ddoc_get()
            assert ddoc['owner'] == self.owner_label()
            self.mongo_handle.db.drivers.update(
                    {'exp_key': self.exp_key},
                    {'$set': {'owner': None}},
                    safe=True, upsert=False, multi=False)
            self._locks = 0
        else:
            raise OperationFailure('no lock to release')

    def exclusive_access(self, force=False):
        class Context(object):
            def __enter__(subself):
                self.ddoc_lock(force=force)
                return subself
            def __exit__(subself, *args):
                self.ddoc_release()
        return Context()

    def ddoc_set_workdir(self):
        """
        """
        with self.exclusive_access() as foo:
            logger.info('setting workdir: %s' % self.workdir)
            ddoc = self.ddoc_get()
            self.mongo_handle.update(ddoc, {'workdir': self.workdir})

    def ddoc_attach_bandit_tuple(self, name, args, kwargs):
        with self.exclusive_access() as foo:
            logger.info('attaching bandit tuple')
            ddoc = self.ddoc_get()
            self.mongo_handle.set_attachment(ddoc,
                        cPickle.dumps((name, args, kwargs)),
                        name='bandit_args_kwargs',
                        collection=self.mongo_handle.db.drivers)

    def save_to_db(self):
        with self.exclusive_access() as foo:
            logger.info('saving state to mongo')
            ddoc = self.ddoc_get()
            self.mongo_handle.set_attachment(ddoc,
                    cPickle.dumps(self),
                    name='pkl',
                    collection=self.mongo_handle.db.drivers)

    def load_from_db(self):
        """
        Return a MongoExperiments instance unpickled from DB

        Raises OperationFailure if no such attachment exists
        """
        logger.info('loading state from mongo')
        ddoc = self.ddoc_get()
        blob = self.mongo_handle.get_attachment(ddoc, name='pkl')
        return cPickle.loads(blob)


class MongoWorker(object):
    poll_interval = 3.0            # seconds
    workdir = None

    def __init__(self, mj,
            poll_interval=poll_interval,
            workdir=workdir):
        self.mj = mj
        self.poll_interval = poll_interval
        self.workdir = workdir

    def run_one(self, host_id=None, reserve_timeout=None):
        if host_id == None:
            host_id = '%s:%i'%(socket.gethostname(), os.getpid()),
        job = None
        start_time = time.time()
        mj = self.mj
        while job is None:
            if (time.time() - start_time) > reserve_timeout:
                raise ReserveTimeout()
            job = mj.reserve(host_id)
            if not job:
                interval = (1 +
                        numpy.random.rand()
                        * (float(self.poll_interval) - 1.0))
                logger.info('no job found, sleeping for %.1fs' % interval)
                time.sleep(interval)

        logger.info('job found: %s' % str(job))

        spec = copy.deepcopy(job['spec']) # don't let the cmd mess up our trial object

        ctrl = CtrlObj(
                jobs=mj,
                read_only=False,
                read_only_id=None,
                )
        ctrl.current_job = job
        config = mj.db.drivers.find_one({'exp_key': job['exp_key']})
        if self.workdir is None:
            workdir = os.path.join(config['workdir'], str(job['_id']))
        else:
            workdir = os.path.expanduser(self.workdir)
        os.makedirs(workdir)
        os.chdir(workdir)
        cmd_protocol = job['cmd'][0]
        try:
            if cmd_protocol == 'cpickled fn':
                worker_fn = cPickle.loads(job['cmd'][1])
            elif cmd_protocol == 'call evaluate':
                bandit = cPickle.loads(job['cmd'][1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'token_load':
                cmd_toks = cmd.split('.')
                cmd_module = '.'.join(cmd_toks[:-1])
                worker_fn = exec_import(cmd_module, cmd)
            elif cmd_protocol == 'bandit_json evaluate':
                bandit = utils.json_call(job['cmd'][1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'db_bandit_construct':
                bandit_str = mj.get_attachment(['bandit_algo_args'])
            else:
                raise ValueError('Unrecognized cmd protocol', cmd_protocol)

            result = worker_fn(spec, ctrl)
            logger.info('job returned: %s' % str(result))
        except Exception, e:
            #TODO: save exception to database, but if this fails, then at least raise the original
            # traceback properly
            logger.info('job exception: %s' % str(e))
            ctrl.checkpoint()
            mj.update(job,
                    {'state': STATE_ERROR,
                    'error': (str(type(e)), str(e))},
                    safe=True)
            raise
        logger.info('job finished: %s' % str(job['_id']))
        ctrl.checkpoint(result)
        mj.update(job, {'state': STATE_DONE}, safe=True)


class CtrlObj(object):
    def __init__(self, read_only=False, read_only_id=None, **kwargs):
        self.current_job = None
        self.__dict__.update(kwargs)
        self.read_only=read_only
        self.read_only_id=read_only_id
        # self.jobs is a MongoJobs reference

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
            if result is not None:
                return self.jobs.update(self.current_job, dict(result=result))


def exec_import(cmd_module, cmd):
    exec('import %s; worker_fn = %s' % (cmd_module, cmd))
    return worker_fn


def as_mongo_str(s):
    if s.startswith('mongo://'):
        return s
    else:
        return 'mongo://%s' % s


def main_worker():
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options]")

    parser.add_option("--max-consecutive-failures",
            dest="max_consecutive_failures",
            metavar='N',
            default=4,
            help="stop if N consecutive jobs fail (default: 4)",
            )
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
    parser.add_option("--workdir",
            dest="workdir",
            default=None,
            help="root workdir (default: load from mongo)",
            metavar="DIR")

    (options, args) = parser.parse_args()

    if args:
        parser.print_help()
        return -1

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
        # XXX: the name of the jobs collection is a parameter elsewhere
        mj = MongoJobs.new_from_connection_str(
                as_mongo_str(options.mongo) + '/jobs')

        md = MongoJobs.new_from_connection_str(
                as_mongo_str(options.mongo) + '/drivers')

        job = None
        while job is None:
            job = mj.reserve(
                    host_id = '%s:%i'%(socket.gethostname(), os.getpid()),
                    )
            if not job:
                interval = (1 +
                        numpy.random.rand()
                        * (float(options.poll_interval) - 1.0))
                logger.info('no job found, sleeping for %.1fs' % interval)
                time.sleep(interval)

        logger.info('job found: %s' % str(job))

        spec = copy.deepcopy(job['spec']) # don't let the cmd mess up our trial object

        ctrl = CtrlObj(
                jobs=mj,
                read_only=False,
                read_only_id=None,
                )
        ctrl.current_job = job
        if options.workdir is None:
            config = mj.db.drivers.find_one({"exp_key": job["exp_key"]})
            workdir = os.path.join(config['workdir'], str(job['_id']))
        else:
            workdir = os.path.expanduser(options.workdir)
        os.makedirs(workdir)
        os.chdir(workdir)
        cmd_protocol = job['cmd'][0]
        try:
            if cmd_protocol == 'cpickled fn':
                worker_fn = cPickle.loads(job['cmd'][1])
            elif cmd_protocol == 'call evaluate':
                bandit = cPickle.loads(job['cmd'][1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'token_load':
                cmd_toks = cmd.split('.')
                cmd_module = '.'.join(cmd_toks[:-1])
                worker_fn = exec_import(cmd_module, cmd)
            elif cmd_protocol == 'bandit_json evaluate':
                bandit = utils.json_call(job['cmd'][1])
                worker_fn = bandit.evaluate
            elif cmd_protocol == 'driver_attachment':
                driver = md.coll.find_one({'exp_key': job['exp_key']})
                blob = md.get_attachment(driver, 'bandit_args_kwargs')
                bandit_name, bandit_args, bandit_kwargs = cPickle.loads(blob)
                worker_fn = utils.json_call(bandit_name,
                        args=bandit_args,
                        kwargs=bandit_kwargs)
            else:
                raise ValueError('Unrecognized cmd protocol', cmd_protocol)

            result = worker_fn(spec, ctrl)
        except Exception, e:
            #TODO: save exception to database, but if this fails, then at least raise the original
            # traceback properly
            ctrl.checkpoint()
            mj.update(job,
                    {'state': STATE_ERROR,
                    'error': (str(type(e)), str(e))},
                    safe=True)
            raise
        ctrl.checkpoint(result)
        mj.update(job, {'state': STATE_DONE}, safe=True)
    else:
        parser.print_help()
        return -1


def main_search():
    from optparse import OptionParser
    parser = OptionParser(
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

    #
    # Construct bandit
    #
    bandit_name = args[0]
    if options.bandit_argfile:
        bandit_argfile_text = open(options.bandit_argfile).read()
        bandit_argv, bandit_kwargs = cPickle.load(bandit_argfile_text)
    else:
        bandit_argfile_text = ''
        bandit_argv, bandit_kwargs = (), {}
    bandit = utils.json_call(bandit_name, bandit_argv, bandit_kwargs)

    #
    # Construct algo
    #
    algo_name = args[1]
    if options.bandit_algo_argfile:
        # in theory this is easy just as above.
        # need tests though, and it's just not done yet.
        raise NotImplementedError('Option: --bandit-algo-argfile')
    else:
        algo_argfile_text = ''
        algo_argv, algo_kwargs = (), {}
    algo = utils.json_call(algo_name, (bandit,) + algo_argv, algo_kwargs)


    #
    # Determine exp_key
    #
    if None is options.exp_key:
        if bandit_argfile_text or algo_argfile_text:
            m = hashlib.md5()
            m.update(bandit_argfile_text)
            m.update(algo_argfile_text)
            exp_key = '%s/%s[arghash:%s]' % (bandit_name, algo_name, m.digest())
            del m
        else:
            exp_key = '%s/%s' % (bandit_name, algo_name)
    else:
        exp_key = options.exp_key

    #
    # Construct MongoExperiment
    #
    if bandit_argfile_text or algo_argfile_text:
        worker_cmd = ('driver_attachment', exp_key)
    else:
        worker_cmd = ('bandit_json evaluate', bandit_name)
    mj = MongoJobs.new_from_connection_str(
            as_mongo_str(options.mongo) + '/jobs')
    self = MongoExperiment(
        bandit_algo=algo,
        mongo_handle=mj,
        workdir=options.workdir,
        exp_key=exp_key,
        poll_interval_secs=(int(options.poll_interval))
            if options.poll_interval else 5,
        max_queue_len=options.max_queue_len,
        cmd=worker_cmd)

    self.ddoc_get()  # init the driver document if necessary, and get it

    # XXX: this is bad, better to check what bandit_tuple is already there
    #      and assert that it matches if something is already there
    self.ddoc_attach_bandit_tuple(bandit_name, bandit_argv, bandit_kwargs)

    if options.clear_existing:
        print >> sys.stdout, "Are you sure you want to delete",
        print >> sys.stdout, ("all %i jobs with exp_key: '%s' ?"
                % (mj.jobs.find({'exp_key':exp_key}).count(),
                    str(exp_key)))
        print >> sys.stdout, '(y/n)'
        y, n = 'y', 'n'
        if input() != 'y':
            print >> sys.stdout, "aborting"
            del self
            return 1

        self.ddoc_lock(force=options.force_lock)
        self.clear_from_db()
        # -- clearing self from db deletes the document used for locking
        #    so no need to release the lock

        # -- re-insert a new driver document
        self.ddoc_get()
        self.ddoc_attach_bandit_tuple(bandit_name, bandit_argv, bandit_kwargs)


    # TODO: uncomment this error when it happens again, and I can see
    # where in the traceback to put it.
    #logger.error('experiment in progress: %s' % str(driver['owner']))
    #logger.error('(hint: run with --force-lock to proceed anyway.)')


    #
    # Try replacing the self we constructed earlier
    # with the unpickled saved state:
    #
    try:
        other = self.load_from_db()
    except OperationFailure:
        pass
    else:
        assert other.exp_key == exp_key
        other.mongo_handle = mj
        other.max_queue_len=self.max_queue_len
        if options.workdir is not None:
            other.workdir = self.workdir
        if options.poll_interval is not None:
            other.poll_interval_secs = self.poll_interval_secs
        self = other

    with self.exclusive_access(force=options.force_lock) as _foo:
        try:
            self.run(options.steps, block_until_done=options.block)
        finally:
            if options.save_on_exit:
                # XXX: does this mess up the original exception traceback?
                try:
                    self.save_to_db()
                except pymongo.errors.OperationFailure:
                    pass


def main_show():
    from optparse import OptionParser
    parser = OptionParser(
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


    #
    # Construct bandit
    #
    # XXX factor this code, shared with main_search
    bandit_name = options.bandit
    if options.bandit_argfile:
        bandit_argfile_text = open(options.bandit_argfile).read()
        bandit_argv, bandit_kwargs = cPickle.load(bandit_argfile_text)
    else:
        bandit_argfile_text = ''
        bandit_argv, bandit_kwargs = (), {}
    bandit = utils.json_call(bandit_name, bandit_argv, bandit_kwargs)

    #
    # Construct algo
    #
    # XXX factor this code, shared with main_search
    algo_name = options.bandit_algo
    if options.bandit_algo_argfile:
        # in theory this is easy just as above.
        # need tests though, and it's just not done yet.
        raise NotImplementedError('Option: --bandit-algo-argfile')
    else:
        algo_argfile_text = ''
        algo_argv, algo_kwargs = (), {}
    algo = utils.json_call(algo_name, (bandit,) + algo_argv, algo_kwargs)


    #
    # Determine exp_key
    #
    # XXX factor this code, shared with main_search
    if None is options.exp_key:
        if bandit_argfile_text or algo_argfile_text:
            m = hashlib.md5()
            m.update(bandit_argfile_text)
            m.update(algo_argfile_text)
            exp_key = '%s/%s[arghash:%s]' % (bandit_name, algo_name, m.digest())
            del m
        else:
            exp_key = '%s/%s' % (bandit_name, algo_name)
    else:
        exp_key = options.exp_key

    mj = MongoJobs.new_from_connection_str(
            as_mongo_str(options.mongo) + '/jobs')
            
    self = MongoExperiment(
        bandit_algo=algo,
        mongo_handle=mj,
        exp_key=exp_key,
        workdir=None,
        poll_interval_secs=0,
        max_queue_len=0,
        cmd=None)
        
    self.refresh_trials_results()

    try:
        other = self.load_from_db()
    except OperationFailure:
        pass
    else:
        assert other.exp_key == exp_key
        other.mongo_handle = mj
        self = other

    if 'history' == cmd:
        import plotting
        import matplotlib.pyplot as plt
        self.refresh_trials_results()
        yvals, colors = zip(*[(1 - r.get('best_epoch_test', .5), 'g')
            for y, r in zip(self.losses(), self.results) if y is not None])
        plt.scatter(range(len(yvals)), yvals, c=colors)
        return plotting.main_plot_history(self)
    elif 'dump' == cmd:
        raise NotImplementedError('TODO: dump jobs db to stdout as JSON')
    elif 'dump_pickle' == cmd:
        self.refresh_trials_results()
        cPickle.dump(self, sys.stdout)
    elif 'vars' == cmd:
        import plotting
        return plotting.main_plot_vars(self)
    else:
        logger.error("Invalid cmd %s" % cmd)
        parser.print_help()
        return -1
