# Parallelizing Evaluations During Search via MongoDB

Hyperopt is designed to support different kinds of trial databases.
The default trial database (`Trials`) is implemented with Python lists and dictionaries.
The default implementation is a reference implementation and it is easy to work with,
but it does not support the asynchronous updates required to evaluate trials in parallel.
For parallel search, hyperopt includes a `MongoTrials` implementation that supports asynchronous updates.

To run a parallelized search, you will need to do the following (after [installing mongodb](Installation-Notes)):

1. Start a mongod process somewhere network-visible.

2. Modify your call to `hyperopt.fmin` to use a MongoTrials backend connected to that mongod process.

3. Start one or more `hyperopt-mongo-worker` processes that will also connect to the mongod process,
    and carry out the search while `fmin` blocks.

## Start a mongod process

Once mongodb is installed, starting a database process (mongod) is as easy as typing e.g.

```bash
mongod --dbpath . --port 1234
# or storing each db its own directory is nice:
mongod --dbpath . --port 1234 --directoryperdb --journal --nohttpinterface
# or consider starting mongod as a daemon:
mongod --dbpath . --port 1234 --directoryperdb --fork --journal --logpath log.log --nohttpinterface
```

Mongo has a habit of pre-allocating a few GB of space (you can disable this with --noprealloc) for better performance, so think a little about where you want to create this database.
Creating a database on a networked filesystem may give terrible performance not only to your database but also to everyone else on your network, be careful about it.

Also, if your machine is visible to the internet, then either bind to the loopback interface and connect via ssh or read mongodb's documentation on password protection.

The rest of the tutorial is based on mongo running on **port 1234** of the **localhost**.

## Use MongoTrials

Suppose, to keep things really simple, that you wanted to minimize the `math.sin` function with hyperopt.
To run things in-process (serially) you could type things out like this:

```python
import math
from hyperopt import fmin, tpe, hp, Trials

trials = Trials()
best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
```

To use the mongo database for persistent storage of the experiment, use a `MongoTrials` object instead of `Trials` like this:

```python
import math
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
```

The first argument to MongoTrials tells it what mongod process to use, and which *database* (here 'foo_db') within that process to use.
The second argument (`exp_key='exp_1'`) is useful for tagging a particular set of trials *within* a database.
The exp_key argument is technically optional.

**N.B.** There is currently an implementation requirement that the database name be followed by '/jobs'.

Whether you always put your trials in separate databases or whether you use the exp_key mechanism to distinguish them is up to you.
In favour of databases: they can be manipulated from the shell (they appear as distinct files) and they ensure greater independence/isolation of experiments.
In favour of exp_key: hyperopt-mongo-worker processes (see below) poll at the database level so they can simultaneously support multiple experiments that are using the same database.

## Run `hyperopt-mongo-worker`

If you run the code fragment above, you will see that it blocks (hangs) at the call fmin.
MongoTrials describes itself internally to fmin as an *asynchronous* trials object, so fmin
does not actually evaluate the objective function when a new search point has been suggested.
Instead, it just sits there, patiently waiting for another process to do that work and update the mongodb with the results.
The `hyperopt-mongo-worker` script included in the `bin` directory of hyperopt was written for this purpose.
It should have been installed on your `$PATH` when you installed hyperopt.

While the `fmin` call in the script above is blocked, open a new shell and type

```bash
hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
```

It will dequeue a work item from the mongodb, evaluate the `math.sin` function, store the results back to the database.
After the `fmin` function has tried enough points it will return and the script above will terminate.
The `hyperopt-mongo-worker` script will then sit around for a few minutes waiting for more work to appear, and then terminate too.

We set the poll interval explicitly in this case because the default timings are set up for jobs (search point evaluations) that take at least a minute or two to complete.

## MongoTrials is a Persistent Object

If you run the example above a second time,

```python
best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
```

you will see that it returns right away and nothing happens.
That's because the database you are connected to already has enough trials in it; you already computed them when you ran the first experiment.
If you want to do another search, you can change the database name or the `exp_key`.
If you want to extend the search, then you can call fmin with a higher number for `max_evals`.
Alternatively, you can launch other processes that create the MongoTrials specifically to analyze the results that are already in the database. Those other processes do not need to call fmin at all.
