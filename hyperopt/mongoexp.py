""" Mongo-based Experiment driver and worker client

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

* config - a collection with a single document, telling workers in what folder
    to work. TODO: have a document per exp_key instead.

* drivers - documents describing drivers. These are used to prevent two drivers
    from using the same exp_key simultaneously, and to attach saved states.

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
bandit.evalute function has returned before setting the status to 2 or 3 to
finalize the job in the database.

"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, Harvard University"
__license__   = "3-clause BSD License"
__contact__   = "hyperopt project at github.com"

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
from bson import BSON, SON

import base
import utils

logger = logging.getLogger(__name__)

# Proxy that could be factored out
# if we also want to use CouchDB
# and JobmanDB classes with this interface
class OperationFailure (Exception):
    pass


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

    def refresh(self, doc, safe=False):
        self.update(doc, dict(refresh_time=coarse_utcnow()), safe=False)

    def update(self, doc, dct, safe=True, collection=None):
        """Return union of doc and dct, after making sure that dct has been
        added to doc in `collection`.

        This function does not modify either `doc` or `dct`.

        safe=True means error-checking is done. safe=False means this function will succeed
        regardless of what happens with the db.
        """
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
            self.coll.update( 
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
            server_doc = self.coll.find_one(
                    dict(_id=doc['_id'], version=doc['version']))
            if server_doc is None:
                raise OperationFailure('updated doc not found : %s'
                        % str(doc))
            elif server_doc != doc:
                if 0:# This is all commented out because it is tripping on the fact that
                    # str('a') != unicode('a').
                    # TODO: eliminate false alarms and catch real ones
                    mismatching_keys = []
                    for k,v in server_doc.items():
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

    def set_attachment(self, doc, blob, name):
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
            doc = self.update(doc, {'_attachments': new_attachments})
            # there is a database leak until we actually delete the files that
            # are no longer pointed to by new_attachments
            ii = 0
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

        Raises KeyError if `name` does not correspond to an attached blob.

        Returns the blob as a string.
        """
        attachments = doc.get('_attachments', [])
        file_ids = [a[1] for a in attachments if a[0] == name]
        if not file_ids:
            raise OperationFailure('Attachment not found: %s' % name)
        if len(file_ids) > 1:
            raise OperationFailure('multiple name matches', (name, file_ids))
        return self.gfs.get(file_ids[0]).read()

    def delete_attachment(self, doc, name):
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
        self.update(doc, {'_attachments':attachments})
        self.gfs.delete(file_id)


class MongoExperiment(base.Experiment):
    """
    This experiment uses a Mongo collection to store
    - self.trials
    - self.results

    """
    def __init__(self, bandit_json, bandit_algo_json, mongo_handle, workdir,
            exp_key, poll_interval_secs = 10):
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
        self.exp_key = exp_key

    def __getstate__(self):
        rval = dict(self.__dict__)
        del rval['mongo_handle']
        return rval

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

    assert 'hyperopt-mongo-worker' in script  # not essential, just debugging here
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
                time.sleep(1 + numpy.random.rand() * 4)

        logger.info('job found: %s' % str(job))

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

def as_mongo_str(s):
    if s.startswith('mongo://'):
        return s
    else:
        return 'mongo://%s' % s

def main_search():
    from optparse import OptionParser
    parser = OptionParser(
            usage="%prog [options] start|resume [bandit bandit_algo]")
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
            help="check work queue every N seconds (default: 10")
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

    (options, args) = parser.parse_args()
    print options, args

    if None is options.exp_key:
        exp_key = args[0] + "/" + args[1]
    else:
        exp_key = options.exp_key

    mj = MongoJobs.new_from_connection_str(
            as_mongo_str(options.mongo) + '/jobs')

    md = MongoJobs.new_from_connection_str(
            as_mongo_str(options.mongo) + '/drivers')

    driver = md.coll.find_one({'exp_key': exp_key})

    if driver and driver['owner'] and not options.force_lock:
        logger.error('experiment in progress: %s' % str(driver['owner']))
        logger.error('(hint: run with --force-lock to proceed anyway.)')
        return -1

    owner = '%s:%i' % (socket.gethostname(), os.getpid())
    if driver is None:
        driver = md.insert(dict(exp_key=exp_key, owner=owner))
    else:
        driver = md.update(driver, dict(owner=owner))

    # checkpoint: we have a driver document, and we are registered as owner
    assert '_id' in driver
    assert driver['owner'] == owner
    assert driver['exp_key'] == exp_key

    try:
        if options.clear_existing:
            # delete any saved driver
            for name in md.attachment_names(driver):
                md.delete_attachment(driver, name)

            # delete any jobs from a previous driver
            mj.delete_all(cond={'exp_key': exp_key})

        if 'pkl' in mj.attachment_names(driver):
            logger.info('loading from saved state')
            blob = mj.get_attachment(driver, 'pkl')
            self = cPickle.loads(blob)
            assert self.exp_key == exp_key
            self.mongo_handle = mj
            if options.workdir is not None:
                self.workdir = options.workdir
            if options.poll_interval is not None:
                self.poll_interval = int(options.poll_interval)
        else:
            logger.info('loading new MongoExperiment')
            self = MongoExperiment(
                bandit_json = args[0],
                bandit_algo_json = args[1],
                mongo_handle = mj,
                workdir = options.workdir,
                exp_key = exp_key,
                poll_interval_secs = (int(options.poll_interval))
                    if options.poll_interval else 10)

        self.run(options.steps)
    finally:
        # if we still have a db connection, unregister ourselves as the active
        # driver

        # XXX: does this mess up the original exception traceback?
        try:
            if 'self' in locals() and options.save_on_exit:
                logger.info('saving state to mongo')
                md.set_attachment(driver, cPickle.dumps(self), name='pkl')
            md.update(driver, dict(owner=None))
        except pymongo.errors.OperationFailure:
            pass

