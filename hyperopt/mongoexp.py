"""
Mongodb-based Trials Object
===========================

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

The MongoJobs, and CtrlObj classes as well as the main_worker
method form the abstraction barrier around this database layout.


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
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
import copy
# import hashlib
import logging
import optparse
import os
# import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import warnings

import numpy
import pymongo
import gridfs
from bson import SON

from .base import JOB_STATES
from .base import (JOB_STATE_NEW, JOB_STATE_RUNNING, JOB_STATE_DONE,
                   JOB_STATE_ERROR)
from .base import Trials
from .base import InvalidTrial
from .base import Ctrl
from .base import SONify
from .base import spec_from_misc
from .utils import coarse_utcnow
from .utils import fast_isin
from .utils import get_most_recent_inds
from .utils import json_call
from .utils import working_dir, temp_dir
import six
from six.moves import map
from six.moves import range

__authors__ = ["James Bergstra", "Dan Yamins"]
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

standard_library.install_aliases()
logger = logging.getLogger(__name__)

try:
    import dill as pickler
except Exception as e:
    logger.info('Failed to load dill, try installing dill via "pip install dill" for enhanced pickling support.')
    import six.moves.cPickle as pickler


class OperationFailure(Exception):
    """Proxy that could be factored out if we also want to use CouchDB and
    JobmanDB classes with this interface
    """


class Shutdown(Exception):
    """
    Exception for telling mongo_worker loop to quit
    """


class WaitQuit(Exception):
    """
    Exception for telling mongo_worker loop to quit
    """


class InvalidMongoTrial(InvalidTrial):
    pass


class DomainSwapError(Exception):
    """Raised when the search program tries to change the bandit attached to
    an experiment.
    """


class ReserveTimeout(Exception):
    """No job was reserved in the alotted time
    """


def read_pw():
    return open(os.path.join(os.getenv('HOME'), ".hyperopt")).read()[:-1]


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

    protocol = url[:url.find(':')]
    ftp_url = 'ftp' + url[url.find(':'):]

    # -- parse the string as if it were an ftp address
    tmp = urllib.parse.urlparse(ftp_url)
    query_params = urllib.parse.parse_qs(tmp.query)

    logger.info('PROTOCOL %s' % protocol)
    logger.info('USERNAME %s' % tmp.username)
    logger.info('HOSTNAME %s' % tmp.hostname)
    logger.info('PORT %s' % tmp.port)
    logger.info('PATH %s' % tmp.path)
    
    authdbname = None
    if 'authSource' in query_params and len(query_params['authSource']):
        authdbname = query_params['authSource'][-1]
        
    logger.info('AUTH DB %s' % authdbname)
    
    try:
        _, dbname, collection = tmp.path.split('/')
    except:
        print("Failed to parse '%s'" % (str(tmp.path)), file=sys.stderr)
        raise
    logger.info('DB %s' % dbname)
    logger.info('COLLECTION %s' % collection)

    if tmp.password is None:
        if (tmp.username is not None) and pwfile:
            password = open(pwfile).read()[:-1]
        else:
            password = None
    else:
        password = tmp.password
    if password is not None:
        logger.info('PASS ***')
    port = int(float(tmp.port))  # port has to be casted explicitly here.

    return (protocol, tmp.username, password, tmp.hostname, port, dbname, collection, authdbname)


def connection_with_tunnel(dbname, host='localhost',
                           auth_dbname=None, port=27017,
                           ssh=False, user='hyperopt', pw=None):
    if ssh:
        local_port = numpy.random.randint(low=27500, high=28000)
        # -- forward from local to remote machine
        ssh_tunnel = subprocess.Popen(
            ['ssh', '-NTf', '-L',
                    '%i:%s:%i' % (local_port, '127.0.0.1', port),
                    host],
        )
        # -- give the subprocess time to set up
        time.sleep(.5)
        connection = pymongo.MongoClient('127.0.0.1', local_port,
                                         document_class=SON, w=1, j=True)
    else:
        connection = pymongo.MongoClient(host, port, document_class=SON, w=1, j=True)
        if user:
            if not pw:
                pw = read_pw()
                
            if user == 'hyperopt' and not auth_dbname:
                auth_dbname = 'admin'
                    
            connection[dbname].authenticate(user, pw, source=auth_dbname)
            
        ssh_tunnel = None

    # Note that the w=1 and j=True args to MongoClient above should:
    # -- Ensure that changes are written to at least one server.
    # -- Ensure that changes are written to the journal if there is one.

    return connection, ssh_tunnel


def connection_from_string(s):
    protocol, user, pw, host, port, db, collection, authdb = parse_url(s)
    if protocol == 'mongo':
        ssh = False
    elif protocol in ('mongo+ssh', 'ssh+mongo'):
        ssh = True
    else:
        raise ValueError('unrecognized protocol for MongoJobs', protocol)
    connection, tunnel = connection_with_tunnel(
        dbname=db,
        ssh=ssh,
        user=user,
        pw=pw,
        host=host,
        port=port,
        auth_dbname=authdb
    )
    return connection, tunnel, connection[db], connection[db][collection]


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
        """
        Parameters
        ----------

        db - Mongo Database (e.g. `Connection()[dbname]`)
            database in which all job-related info is stored

        jobs - Mongo Collection handle
            collection within `db` to use for job arguments, return vals,
            and various bookkeeping stuff and meta-data. Typically this is
            `db['jobs']`

        gfs - Mongo GridFS handle
            GridFS is used to store attachments - binary blobs that don't fit
            or are awkward to store in the `jobs` collection directly.

        conn - Mongo Connection
            Why we need to keep this, I'm not sure.

        tunnel - something for ssh tunneling if you're doing that
            See `connection_with_tunnel` for more info.

        config_name - string
            XXX: No idea what this is for, seems unimportant.

        """
        self.db = db
        self.jobs = jobs
        self.gfs = gfs
        self.conn = conn
        self.tunnel = tunnel
        self.config_name = config_name

    # TODO: rename jobs -> coll throughout
    coll = property(lambda s: s.jobs)

    @classmethod
    def alloc(cls, dbname, host='localhost',
              auth_dbname='admin', port=27017,
              jobs_coll='jobs', gfs_coll='fs', ssh=False, user=None, pw=None):
        connection, tunnel = connection_with_tunnel(
            dbname, host, auth_dbname, port, ssh, user, pw)
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
        c = self.jobs.find(filter=dict(state=JOB_STATE_DONE))
        return c if cursor else list(c)

    def jobs_error(self, cursor=False):
        c = self.jobs.find(filter=dict(state=JOB_STATE_ERROR))
        return c if cursor else list(c)

    def jobs_running(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(filter=dict(state=JOB_STATE_RUNNING)))
        # TODO: mark some as MIA
        rval = [r for r in rval if not r.get('MIA', False)]
        return rval

    def jobs_dead(self, cursor=False):
        if cursor:
            raise NotImplementedError()
        rval = list(self.jobs.find(filter=dict(state=JOB_STATE_RUNNING)))
        # TODO: mark some as MIA
        rval = [r for r in rval if r.get('MIA', False)]
        return rval

    def jobs_queued(self, cursor=False):
        c = self.jobs.find(filter=dict(state=JOB_STATE_NEW))
        return c if cursor else list(c)

    def insert(self, job):
        """Return a job dictionary by inserting the job dict into the database"""
        try:
            cpy = copy.deepcopy(job)
            # -- this call adds an _id field to cpy
            _id = self.jobs.insert(cpy, check_keys=True)
            # -- so now we return the dict with the _id field
            assert _id == cpy['_id']
            return cpy
        except pymongo.errors.OperationFailure as e:
            # -- translate pymongo error class into hyperopt error class
            #    This was meant to make it easier to catch insertion errors
            #    in a generic way even if different databases were used.
            #    ... but there's just MongoDB so far, so kinda goofy.
            raise OperationFailure(e)

    def delete(self, job):
        """Delete job[s]"""
        try:
            self.jobs.remove(job)
        except pymongo.errors.OperationFailure as e:
            # -- translate pymongo error class into hyperopt error class
            #    see insert() code for rationale.
            raise OperationFailure(e)

    def delete_all(self, cond=None):
        """Delete all jobs and attachments"""
        if cond is None:
            cond = {}
        try:
            for d in self.jobs.find(filter=cond, projection=['_id', '_attachments']):
                logger.info('deleting job %s' % d['_id'])
                for name, file_id in d.get('_attachments', []):
                    try:
                        self.gfs.delete(file_id)
                    except gridfs.errors.NoFile:
                        logger.error('failed to remove attachment %s:%s' % (
                            name, file_id))
                self.jobs.remove(d)
        except pymongo.errors.OperationFailure as e:
            # -- translate pymongo error class into hyperopt error class
            #    see insert() code for rationale.
            raise OperationFailure(e)

    def delete_all_error_jobs(self):
        return self.delete_all(cond={'state': JOB_STATE_ERROR})

    def reserve(self, host_id, cond=None, exp_key=None):
        now = coarse_utcnow()
        if cond is None:
            cond = {}
        else:
            cond = copy.copy(cond)  # copy is important, will be modified, but only the top-level

        if exp_key is not None:
            cond['exp_key'] = exp_key

        # having an owner of None implies state==JOB_STATE_NEW, so this effectively
        # acts as a filter to make sure that only new jobs get reserved.
        if cond.get('owner') is not None:
            raise ValueError('refusing to reserve owned job')
        else:
            cond['owner'] = None
            cond['state'] = JOB_STATE_NEW  # theoretically this is redundant, theoretically

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
                upsert=False)
        except pymongo.errors.OperationFailure as e:
            logger.error('Error during reserve_job: %s' % str(e))
            rval = None
        return rval

    def refresh(self, doc):
        self.update(doc, dict(refresh_time=coarse_utcnow()))

    def update(self, doc, dct, collection=None, do_sanity_checks=True):
        """Return union of doc and dct, after making sure that dct has been
        added to doc in `collection`.

        This function does not modify either `doc` or `dct`.

        """
        if collection is None:
            collection = self.coll

        dct = copy.deepcopy(dct)
        if '_id' not in doc:
            raise ValueError('doc must have an "_id" key to be updated')

        if '_id' in dct:
            if dct['_id'] != doc['_id']:
                raise ValueError('cannot update the _id field')
            del dct['_id']

        if 'version' in dct:
            if dct['version'] != doc['version']:
                warnings.warn('Ignoring "version" field in update dictionary')

        if 'version' in doc:
            doc_query = dict(_id=doc['_id'], version=doc['version'])
            dct['version'] = doc['version'] + 1
        else:
            doc_query = dict(_id=doc['_id'])
            dct['version'] = 1
        try:
            # warning - if doc matches nothing then this function succeeds
            # N.B. this matches *at most* one entry, and possibly zero
            collection.update(
                doc_query,
                {'$set': dct},
                upsert=False,
                multi=False,)
        except pymongo.errors.OperationFailure as e:
            # -- translate pymongo error class into hyperopt error class
            #    see insert() code for rationale.
            raise OperationFailure(e)

        # update doc in-place to match what happened on the server side
        doc.update(dct)

        if do_sanity_checks:
            server_doc = collection.find_one(
                dict(_id=doc['_id'], version=doc['version']))
            if server_doc is None:
                raise OperationFailure('updated doc not found : %s'
                                       % str(doc))
            elif server_doc != doc:
                if 0:  # This is all commented out because it is tripping on the fact that
                    # str('a') != unicode('a').
                    # TODO: eliminate false alarms and catch real ones
                    mismatching_keys = []
                    for k, v in list(server_doc.items()):
                        if k in doc:
                            if doc[k] != v:
                                mismatching_keys.append((k, v, doc[k]))
                        else:
                            mismatching_keys.append((k, v, '<missing>'))
                    for k, v in list(doc.items()):
                        if k not in server_doc:
                            mismatching_keys.append((k, '<missing>', v))

                    raise OperationFailure('local and server doc documents are out of sync: %s' %
                                           repr((doc, server_doc, mismatching_keys)))
        return doc

    def attachment_names(self, doc):
        def as_str(name_id):
            assert isinstance(name_id[0], six.string_types), name_id
            return str(name_id[0])
        return list(map(as_str, doc.get('_attachments', [])))

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

        new_attachments = ([a for a in attachments if a[0] != name] +
                           [(name, new_file_id)])

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
        # return new_file_id

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
        for i, a in enumerate(attachments):
            if a[0] == name:
                file_id = a[1]
                break
        if file_id is None:
            raise OperationFailure('Attachment not found: %s' % name)
        del attachments[i]
        self.update(doc, {'_attachments': attachments}, collection=collection)
        self.gfs.delete(file_id)


class MongoTrials(Trials):
    """Trials maps on to an entire mongo collection. It's basically a wrapper
    around MongoJobs for now.

    As a concession to performance, this object permits trial filtering based
    on the exp_key, but I feel that's a hack. The case of `cmd` is similar--
    the exp_key and cmd are semantically coupled.

    WRITING TO THE DATABASE
    -----------------------
    The trials object is meant for *reading* a trials database. Writing
    to a database is different enough from writing to an in-memory
    collection that no attempt has been made to abstract away that
    difference.  If you want to update the documents within
    a MongoTrials collection, then retrieve the `.handle` attribute (a
    MongoJobs instance) and use lower-level methods, or pymongo's
    interface directly.  When you are done writing, call refresh() or
    refresh_tids() to bring the MongoTrials up to date.
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

    def refresh_tids(self, tids):
        """ Sync documents with `['tid']` in the list of `tids` from the
        database (not *to* the database).

        Local trial documents whose tid is not in `tids` are not
        affected by this call.  Local trial documents whose tid is in `tids` may
        be:

        * *deleted* (if db no longer has corresponding document), or
        * *updated* (if db has an updated document) or,
        * *left alone* (if db document matches local one).

        Additionally, if the db has a matching document, but there is no
        local trial with a matching tid, then the db document will be
        *inserted* into the local collection.

        """
        exp_key = self._exp_key
        if exp_key != None:
            query = {'exp_key': exp_key}
        else:
            query = {}
        t0 = time.time()
        query['state'] = {'$ne': JOB_STATE_ERROR}
        if tids is not None:
            query['tid'] = {'$in': list(tids)}
        orig_trials = getattr(self, '_trials', [])
        _trials = orig_trials[:]  # copy to make sure it doesn't get screwed up
        if _trials:
            db_data = list(self.handle.jobs.find(query,
                                                 projection=['_id', 'version']))
            # -- pull down a fresh list of ids from mongo
            if db_data:
                # make numpy data arrays
                db_data = numpy.rec.array([(x['_id'], int(x['version']))
                                           for x in db_data],
                                          names=['_id', 'version'])
                db_data.sort(order=['_id', 'version'])
                db_data = db_data[get_most_recent_inds(db_data)]

                existing_data = numpy.rec.array([(x['_id'],
                                                  int(x['version'])) for x in _trials],
                                                names=['_id', 'version'])
                existing_data.sort(order=['_id', 'version'])

                # which records are in db but not in existing, and vice versa
                db_in_existing = fast_isin(db_data['_id'], existing_data['_id'])
                existing_in_db = fast_isin(existing_data['_id'], db_data['_id'])

                # filtering out out-of-date records
                _trials = [_trials[_ind] for _ind in existing_in_db.nonzero()[0]]

                # new data is what's in db that's not in existing
                new_data = db_data[numpy.invert(db_in_existing)]

                # having removed the new and out of data data,
                # concentrating on data in db and existing for state changes
                db_data = db_data[db_in_existing]
                existing_data = existing_data[existing_in_db]
                try:
                    assert len(db_data) == len(existing_data)
                    assert (existing_data['_id'] == db_data['_id']).all()
                    assert (existing_data['version'] <= db_data['version']).all()
                except:
                    reportpath = os.path.join(os.getcwd(),
                                              'hyperopt_refresh_crash_report_' +
                                              str(numpy.random.randint(1e8)) + '.pkl')
                    logger.error('HYPEROPT REFRESH ERROR: writing error file to %s' % reportpath)
                    _file = open(reportpath, 'w')
                    pickler.dump({'db_data': db_data,
                                 'existing_data': existing_data},
                                _file)
                    _file.close()
                    raise

                same_version = existing_data['version'] == db_data['version']
                _trials = [_trials[_ind] for _ind in same_version.nonzero()[0]]
                version_changes = existing_data[numpy.invert(same_version)]

                # actually get the updated records
                update_ids = new_data['_id'].tolist() + version_changes['_id'].tolist()
                num_new = len(update_ids)
                update_query = copy.deepcopy(query)
                update_query['_id'] = {'$in': update_ids}
                updated_trials = list(self.handle.jobs.find(update_query))
                _trials.extend(updated_trials)
            else:
                num_new = 0
                _trials = []
        else:
            # this case is for performance, though should be able to be removed
            # without breaking correctness.
            _trials = list(self.handle.jobs.find(query))
            if _trials:
                _trials = [_trials[_i] for _i in get_most_recent_inds(_trials)]
            num_new = len(_trials)

        logger.debug('Refresh data download took %f seconds for %d ids' %
                     (time.time() - t0, num_new))

        if tids is not None:
            # -- If tids were given, then _trials only contains
            #    documents with matching tids. Here we augment these
            #    fresh matching documents, with our current ones whose
            #    tids don't match.
            new_trials = _trials
            tids_set = set(tids)
            assert all(t['tid'] in tids_set for t in new_trials)
            old_trials = [t for t in orig_trials if t['tid'] not in tids_set]
            _trials = new_trials + old_trials

        # -- reassign new trials to self, in order of increasing tid
        jarray = numpy.array([j['_id'] for j in _trials])
        jobsort = jarray.argsort()
        self._trials = [_trials[_idx] for _idx in jobsort]
        self._specs = [_trials[_idx]['spec'] for _idx in jobsort]
        self._results = [_trials[_idx]['result'] for _idx in jobsort]
        self._miscs = [_trials[_idx]['misc'] for _idx in jobsort]

    def refresh(self):
        self.refresh_tids(None)

    def _insert_trial_docs(self, docs):
        rval = []
        for doc in docs:
            rval.append(self.handle.jobs.insert(doc))
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

    def delete_all(self, cond=None):
        if cond is None:
            cond = {}
        else:
            cond = dict(cond)

        if self._exp_key:
            cond['exp_key'] = self._exp_key
        # -- remove all documents matching condition
        self.handle.delete_all(cond)
        gfs = self.handle.gfs
        for filename in gfs.list():
            try:
                fdoc = gfs.get_last_version(filename=filename, **cond)
            except gridfs.errors.NoFile:
                continue
            gfs.delete(fdoc._id)
        self.refresh()

    def new_trial_ids(self, N):
        db = self.handle.db
        # N.B. that the exp key is *not* used here. It was once, but it caused
        # a nasty bug: tids were generated by a global experiment
        # with exp_key=None, running a suggest() that introduced sub-experiments
        # with exp_keys, which ran jobs that did result injection.  The tids of
        # injected jobs were sometimes unique within an experiment, and
        # sometimes not. Hilarious!
        #
        # Solution: tids are generated to be unique across the db, not just
        # within an exp_key.
        #

        # -- mongo docs say you can't upsert an empty document
        query = {'a': 0}

        doc = None
        while doc is None:
            doc = db.job_ids.find_and_modify(
                query,
                {'$inc': {'last_id': N}},
                upsert=True)
            if doc is None:
                logger.warning('no last_id found, re-trying')
                time.sleep(1.0)
        lid = doc.get('last_id', 0)
        return list(range(lid, lid + N))

    def trial_attachments(self, trial):
        """
        Attachments to a single trial (e.g. learned weights)

        Returns a dictionary interface to the attachments.
        """

        # don't offer more here than in MongoCtrl
        class Attachments(object):

            def __contains__(_self, name):
                return name in self.handle.attachment_names(doc=trial)

            def __len__(_self):
                return len(self.handle.attachment_names(doc=trial))

            def __iter__(_self):
                return iter(self.handle.attachment_names(doc=trial))

            def __getitem__(_self, name):
                try:
                    return self.handle.get_attachment(
                        doc=trial,
                        name=name)
                except OperationFailure:
                    raise KeyError(name)

            def __setitem__(_self, name, value):
                self.handle.set_attachment(
                    doc=trial,
                    blob=value,
                    name=name,
                    collection=self.handle.db.jobs)

            def __delitem__(_self, name):
                raise NotImplementedError('delete trial_attachment')

            def keys(self):
                return [k for k in self]

            def values(self):
                return [self[k] for k in self]

            def items(self):
                return [(k, self[k]) for k in self]

        return Attachments()

    @property
    def attachments(self):
        """
        Attachments to a Trials set (such as bandit args).

        Support syntax for load:  self.attachments[name]
        Support syntax for store: self.attachments[name] = value
        """
        gfs = self.handle.gfs

        query = {}
        if self._exp_key:
            query['exp_key'] = self._exp_key

        class Attachments(object):

            def __iter__(_self):
                if query:
                    # -- gfs.list does not accept query kwargs
                    #    (at least, as of pymongo 2.4)
                    filenames = [fname
                                 for fname in gfs.list()
                                 if fname in _self]
                else:
                    filenames = gfs.list()
                return iter(filenames)

            def __contains__(_self, name):
                return gfs.exists(filename=name, **query)

            def __getitem__(_self, name):
                try:
                    rval = gfs.get_version(filename=name, **query).read()
                    return rval
                except gridfs.NoFile:
                    raise KeyError(name)

            def __setitem__(_self, name, value):
                if gfs.exists(filename=name, **query):
                    gout = gfs.get_last_version(filename=name, **query)
                    gfs.delete(gout._id)
                gfs.put(value, filename=name, encoding='utf-8', **query)

            def __delitem__(_self, name):
                gout = gfs.get_last_version(filename=name, **query)
                gfs.delete(gout._id)

        return Attachments()


class MongoWorker(object):
    poll_interval = 3.0  # -- seconds
    workdir = None

    def __init__(self, mj,
                 poll_interval=poll_interval,
                 workdir=workdir,
                 exp_key=None,
                 logfilename='logfile.txt',
                 ):
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
        self.logfilename = logfilename

    def make_log_handler(self):
        self.log_handler = logging.FileHandler(self.logfilename)
        self.log_handler.setFormatter(
            logging.Formatter(
                fmt='%(levelname)s (%(name)s): %(message)s'))
        self.log_handler.setLevel(logging.INFO)

    def run_one(self,
                host_id=None,
                reserve_timeout=None,
                erase_created_workdir=False,
                ):
        if host_id == None:
            host_id = '%s:%i' % (socket.gethostname(), os.getpid()),
        job = None
        start_time = time.time()
        mj = self.mj
        while job is None:
            if (reserve_timeout and
               (time.time() - start_time) > reserve_timeout):
                raise ReserveTimeout()
            job = mj.reserve(host_id, exp_key=self.exp_key)
            if not job:
                interval = (1 +
                            numpy.random.rand() *
                            (float(self.poll_interval) - 1.0))
                logger.info('no job found, sleeping for %.1fs' % interval)
                time.sleep(interval)

        logger.debug('job found: %s' % str(job))

        # -- don't let the cmd mess up our trial object
        spec = spec_from_misc(job['misc'])

        ctrl = MongoCtrl(
            trials=MongoTrials(mj, exp_key=job['exp_key'], refresh=False),
            read_only=False,
            current_trial=job)
        if self.workdir is None:
            workdir = job['misc'].get('workdir', os.getcwd())
            if workdir is None:
                workdir = ''
            workdir = os.path.join(workdir, str(job['_id']))
        else:
            workdir = self.workdir
        workdir = os.path.abspath(os.path.expanduser(workdir))
        try:
            root_logger = logging.getLogger()
            if self.logfilename:
                self.make_log_handler()
                root_logger.addHandler(self.log_handler)

            cmd = job['misc']['cmd']
            cmd_protocol = cmd[0]
            try:
                if cmd_protocol == 'cpickled fn':
                    worker_fn = pickler.loads(cmd[1])
                elif cmd_protocol == 'call evaluate':
                    bandit = pickler.loads(cmd[1])
                    worker_fn = bandit.evaluate
                elif cmd_protocol == 'token_load':
                    cmd_toks = cmd[1].split('.')
                    cmd_module = '.'.join(cmd_toks[:-1])
                    worker_fn = exec_import(cmd_module, cmd[1])
                elif cmd_protocol == 'bandit_json evaluate':
                    bandit = json_call(cmd[1])
                    worker_fn = bandit.evaluate
                elif cmd_protocol == 'driver_attachment':
                    # name = 'driver_attachment_%s' % job['exp_key']
                    blob = ctrl.trials.attachments[cmd[1]]
                    bandit_name, bandit_args, bandit_kwargs = pickler.loads(blob)
                    worker_fn = json_call(bandit_name,
                                          args=bandit_args,
                                          kwargs=bandit_kwargs).evaluate
                elif cmd_protocol == 'domain_attachment':
                    blob = ctrl.trials.attachments[cmd[1]]
                    try:
                        domain = pickler.loads(blob)
                    except BaseException as e:
                        logger.info(
                            'Error while unpickling.')
                        raise
                    worker_fn = domain.evaluate
                else:
                    raise ValueError('Unrecognized cmd protocol', cmd_protocol)

                with temp_dir(workdir, erase_created_workdir), working_dir(workdir):
                    result = worker_fn(spec, ctrl)
                    result = SONify(result)
            except BaseException as e:
                # XXX: save exception to database, but if this fails, then
                #      at least raise the original traceback properly
                logger.info('job exception: %s' % str(e))
                ctrl.checkpoint()
                mj.update(job,
                          {'state': JOB_STATE_ERROR,
                           'error': (str(type(e)), str(e))})
                raise
        finally:
            if self.logfilename:
                root_logger.removeHandler(self.log_handler)

        logger.info('job finished: %s' % str(job['_id']))
        attachments = result.pop('attachments', {})
        for aname, aval in list(attachments.items()):
            logger.info(
                'mongoexp: saving attachment name=%s (%i bytes)' % (
                    aname, len(aval)))
            ctrl.attachments[aname] = aval
        ctrl.checkpoint(result)
        mj.update(job, {'state': JOB_STATE_DONE})


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
        return self.trials.trial_attachments(trial=self.current_trial)

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
    if options.last_job_timeout is not None:
        last_job_timeout = time.time() + float(options.last_job_timeout)
    else:
        last_job_timeout = None

    def sighandler_shutdown(signum, frame):
        logger.info('Caught signal %i, shutting down.' % signum)
        raise Shutdown(signum)

    def sighandler_wait_quit(signum, frame):
        logger.info('Caught signal %i, shutting down.' % signum)
        raise WaitQuit(signum)

    signal.signal(signal.SIGINT, sighandler_shutdown)
    signal.signal(signal.SIGHUP, sighandler_shutdown)
    signal.signal(signal.SIGTERM, sighandler_shutdown)
    signal.signal(signal.SIGUSR1, sighandler_wait_quit)

    if N > 1:
        proc = None
        cons_errs = 0
        if last_job_timeout and time.time() > last_job_timeout:
            logger.info("Exiting due to last_job_timeout")
            return

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
                            '--mongo=%s' % options.mongo,
                            '--reserve-timeout=%s' % options.reserve_timeout]
                if options.workdir is not None:
                    sub_argv.append('--workdir=%s' % options.workdir)
                if options.exp_key is not None:
                    sub_argv.append('--exp-key=%s' % options.exp_key)
                proc = subprocess.Popen(sub_argv)
                retcode = proc.wait()
                proc = None

            except Shutdown:
                # this is the normal way to stop the infinite loop (if originally N=-1)
                if proc:
                    # proc.terminate() is only available as of 2.6
                    os.kill(proc.pid, signal.SIGTERM)
                    return proc.wait()
                else:
                    return 0

            except WaitQuit:
                # -- sending SIGUSR1 to a looping process will cause it to
                # break out of the loop after the current subprocess finishes
                # normally.
                if proc:
                    return proc.wait()
                else:
                    return 0

            if retcode != 0:
                cons_errs += 1
            else:
                cons_errs = 0
            N -= 1
        logger.info("exiting with N=%i after %i consecutive exceptions" % (
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
        raise ValueError("N <= 0")


def main_worker():
    parser = optparse.OptionParser(usage="%prog [options]")

    parser.add_option("--exp-key",
                      dest='exp_key',
                      default=None,
                      metavar='str',
                      help="identifier for this workers's jobs")
    parser.add_option("--last-job-timeout",
                      dest='last_job_timeout',
                      metavar='T',
                      default=None,
                      help="Do not reserve a job after T seconds have passed")
    parser.add_option("--max-consecutive-failures",
                      dest="max_consecutive_failures",
                      metavar='N',
                      default=4,
                      help="stop if N consecutive jobs fail (default: 4)")
    parser.add_option("--max-jobs",
                      dest='max_jobs',
                      default=sys.maxsize,
                      help="stop after running this many jobs (default: inf)")
    parser.add_option("--mongo",
                      dest='mongo',
                      default='localhost/hyperopt',
                      help="<host>[:port]/<db> for IPC and job storage")
    parser.add_option("--poll-interval",
                      dest='poll_interval',
                      metavar='N',
                      default=5,
                      help="check work queue every 1 < T < N seconds (default: 5")
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
