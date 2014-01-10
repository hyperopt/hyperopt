import copy
import cPickle
import pymongo as pm
from shovel import task
import hyperopt
from hyperopt.mongoexp import MongoTrials

@task
def show_vars(host, port, dbname, key, colorize=-1, columns=5):
    """
    Show loss vs. time scatterplots for one experiment or all experiments in a
    database.
    """
    conn = pm.Connection(host=host, port=int(port))
    K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
    for k in K:
        print k
    if key is None:
        raise NotImplementedError()
    else:
        docs = list(
                conn[dbname]['jobs'].find(
                    {'exp_key': key},
                    {
                        'tid': 1,
                        'state': 1,
                        'result.loss':1,
                        'result.status':1,
                        'spec':1,
                        'misc.cmd': 1,
                        'misc.tid':1,
                        'misc.idxs':1,
                        'misc.vals': 1,
                    }))
    doc0 = docs[0]
    cmd = doc0['misc']['cmd']
    print 'cmd:', cmd
    if cmd[0] == 'bandit_json evaluate':
        bandit = hyperopt.utils.json_call(cmd[1])
    elif cmd[0] == 'domain_attachment':
        trials = MongoTrials('mongo://%s:%s/%s/jobs' % (
            host, port, dbname))
        blob = trials.attachments[cmd[1]]
        domain = cPickle.loads(blob)
        bandit = domain
    else:
        raise NotImplementedError('loading bandit from cmd', cmd)

    trials = hyperopt.trials_from_docs(docs, validate=False)
    hyperopt.plotting.main_plot_vars(trials, bandit=bandit,
            colorize_best=int(colorize),
            columns=int(columns)
            )


@task
def show_runtime(host, port, dbname, key):
    """
    Show runtime vs. trial_id
    """
    conn = pm.Connection(host=host, port=int(port))
    K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
    print 'Experiments in database', dbname
    for k in K:
        print '* ', k
    docs = list(
            conn[dbname]['jobs'].find(
                {'exp_key': key},
                {
                    'tid': 1,
                    'state': 1,
                    'book_time': 1,
                    'refresh_time': 1,
                    #'result.loss':1,
                    #'result.status':1,
                    #'spec':1,
                    #'misc.cmd': 1,
                    #'misc.tid':1,
                    #'misc.idxs':1,
                    #'misc.vals': 1,
                }))
    import matplotlib.pyplot as plt
    for state in hyperopt.JOB_STATES:
        x = [d['tid'] for d in docs if d['state'] == state]
        if state != hyperopt.JOB_STATE_NEW:
            y = [(d['refresh_time'] - d['book_time']).total_seconds() / 60.0
                 for d in docs if d['state'] == state]
        else:
            y = [0] * len(x)

        if x:
            plt.scatter(x, y,
                    c=['g', 'b', 'k', 'r'][state],
                    label=['NEW', 'RUNNING', "DONE", "ERROR"][state])
    plt.ylabel('runtime (minutes)')
    plt.xlabel('trial identifier')
    plt.legend(loc='upper left')
    plt.show()


@task
def list_dbs(host, port, dbname=None):
    """
    List the databases and experiments being hosted by a mongo server
    """
    conn = pm.Connection(host=host, port=int(port))
    if dbname is None:
        dbnames = conn.database_names()
    else:
        dbnames = [dbname]

    for dbname in dbnames:
        K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
        print ''
        print 'Database:', dbname
        for k in K:
            print ' ', k


@task
def transfer_trials(fromdb, todb):
    """
    Insert all of the documents in `fromdb` into `todb`.
    """
    from_trials = MongoTrials('mongo://localhost:44556/%s/jobs' % fromdb)
    to_trials = MongoTrials('mongo://localhost:44556/%s/jobs' % todb)
    from_docs = [copy.deepcopy(doc) for doc in from_trials]
    for doc in from_docs:
        del doc['_id']
    to_trials.insert_trial_docs(doc)


@task
def list_all(host, port, dbname, key=None, spec=0, fields=None):
    conn = pm.Connection(host=host, port=int(port))
    if key is None:
        query = {}
    else:
        query = {'exp_key': key}
    if fields is None:
        cursor = conn[dbname]['jobs'].find(query)
    else:
        fields = fields.split(',')
        cursor = conn[dbname]['jobs'].find(query, fields)

    for doc in cursor:
        print repr(doc)


@task
def list_attachments(host, port, dbname, key=None, tid=None):
    if tid is not None:
        tid = int(tid)
        conn = pm.Connection(host=host, port=int(port))
        query = {'tid': tid}
        for doc in conn[dbname]['jobs'].find(query, ['result', '_attachments']):
            for name, ptr in doc['_attachments']:
                print name, ptr
    elif key is not None:
        raise NotImplementedError()
    else:
        raise NotImplementedError()



@task
def list_errors(host, port, dbname, key=None, spec=0):
    conn = pm.Connection(host=host, port=int(port))
    if key is None:
        query = {}
    else:
        query = {'exp_key': key}
    query['state'] = hyperopt.JOB_STATE_ERROR
    retrieve = {'tid': 1, 'state': 1, 'result.status':1, 'misc.cmd': 1,
                'book_time': 1, 'error': 1, 'owner': 1}
    if int(spec):
        retrieve['spec'] = 1
    for doc in conn[dbname]['jobs'].find(query, retrieve):
        print doc['_id'], doc['tid'], doc['book_time'], doc['owner'], doc['error']
        if int(spec):
            print doc['spec']


@task
def list_failures(host, port, dbname, key=None, spec=0):
    conn = pm.Connection(host=host, port=int(port))
    if key is None:
        query = {}
    else:
        query = {'exp_key': key}
    query['state'] = hyperopt.JOB_STATE_DONE
    query['result.status'] = hyperopt.STATUS_FAIL
    retrieve = {'tid': 1,
            'result.failure':1,
            'misc.cmd': 1, 'book_time': 1, 'owner': 1}
    if int(spec):
        retrieve['spec'] = 1
    for doc in conn[dbname]['jobs'].find(query, retrieve):
        print doc['_id'], doc['tid'], doc['book_time'], doc['owner'],
        print doc['result']['failure']
        if int(spec):
            print doc['spec']


@task
def list_best(host, port, dbname, key=None):
    db = 'mongo://%s:%s/%s/jobs' % (host, port, dbname)
    mongo_trials = MongoTrials(db, exp_key=key)
    docs = mongo_trials.trials
    docs = [(d['result']['loss'], d)
            for d in mongo_trials.trials
            if d['state'] == 2 and d['result']['status'] == 'ok']
    docs.sort()
    for loss, doc in reversed(docs):
        print loss, doc['owner'], doc['result']
    print 'total oks:', len(docs)
    print 'total trials:', len(mongo_trials.trials)



@task
def delete_trials(host, port, dbname, key=None):
    y, n = 'y', 'n'
    db = 'mongo://%s:%s/%s/jobs' % (host, port, dbname)
    if key is None:
        mongo_trials = MongoTrials(db)
        n_trials = mongo_trials.handle.jobs.find({}).count()
        print ('Are you sure you want to delete ALL %i trials from %s? (y/n)'
                % (n_trials, db))
        if input() != y:
            print 'Aborting'
            return
        mongo_trials.delete_all()
    else:
        mongo_trials = MongoTrials(db, exp_key=key)
        n_trials = mongo_trials.handle.jobs.find({'exp_key': key}).count()
        print 'Confirm: delete %i trials matching %s? (y/n)' % (
            n_trials, key)
        if input() != y:
            print 'Aborting'
            return
        mongo_trials.delete_all()


@task
def delete_evaltimeout_trials(host, port, dbname, key=None):
    db = 'mongo://%s:%s/%s/jobs' % (host, port, dbname)
    if key is None:
        mongo_trials = MongoTrials(db)
        for t in mongo_trials.trials:
            try:
                if 'EvalTimeout' in str(t['result']['failure']):
                    #del_tid = t['tid']
                    pass
                else:
                    continue
            except KeyError:
                continue
            print t['tid'], t['result']['failure']
            mongo_trials.delete_all(cond={'tid': t['tid']})
    else:
        raise NotImplementedError()


@task
def delete_running(host, port, dbname, key=None):
    db = 'mongo://%s:%s/%s/jobs' % (host, port, dbname)
    if key is None:
        mongo_trials = MongoTrials(db)
    else:
        mongo_trials = MongoTrials(db, exp_key=key)
    mongo_trials.delete_all(cond={'state': 1})


@task
def delete_errors(host, port, dbname, key=None):
    db = 'mongo://%s:%s/%s/jobs' % (host, port, dbname)
    if key is None:
        mongo_trials = MongoTrials(db)
    else:
        mongo_trials = MongoTrials(db, exp_key=key)
    mongo_trials.delete_all(cond={'state': 3})

@task
def snapshot(dbname, ofilename=None, plevel=-1):
    """
    Save the trials of a mongo database to a pickle file.
    """
    print 'fetching trials'
    from_trials = MongoTrials('mongo://honeybadger.rowland.org:44556/%s/jobs' % dbname)
    to_trials = hyperopt.base.trials_from_docs(
            from_trials.trials,
            # -- effing slow, but required for un-pickling??
            validate=True)
    if ofilename is None:
        ofilename = dbname+'.snapshot.pkl'
    print 'saving to', ofilename
    ofile = open(ofilename, 'w')
    cPickle.dump(to_trials, ofile, int(plevel))


