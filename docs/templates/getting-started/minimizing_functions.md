# Defining a Function to Minimize

Hyperopt provides a few levels of increasing flexibility / complexity when it comes to specifying an objective function to minimize.
The questions to think about as a designer are

* Do you want to save additional information beyond the function return value, such as other statistics and diagnostic information collected during the computation of the objective?
* Do you want to use optimization algorithms that require more than the function value?
* Do you want to communicate between parallel processes? (e.g. other workers, or the minimization algorithm)

The next few sections will look at various ways of implementing an objective
function that minimizes a quadratic objective function over a single variable.
In each section, we will be searching over a bounded range from -10 to +10,
which we can describe with a *search space*:

```python
space = hp.uniform('x', -10, 10)
```

Below, Section 2, covers how to specify search spaces that are more complicated.

## The Simplest Case

The simplest protocol for communication between hyperopt's optimization
algorithms and your objective function, is that your objective function
receives a valid point from the search space, and returns the floating-point
*loss* (aka negative utility) associated with that point.

```python
from hyperopt import fmin, tpe, hp
best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)
print(best)
```

This protocol has the advantage of being extremely readable and quick to
type. As you can see, it's nearly a one-liner.
The disadvantages of this protocol are
(1) that this kind of function cannot return extra information about each evaluation into the trials database,
and
(2) that this kind of function cannot interact with the search algorithm or other concurrent function evaluations.
You will see in the next examples why you might want to do these things.

## Attaching Extra Information via the Trials Object

If your objective function is complicated and takes a long time to run, you will almost certainly want to save more statistics
and diagnostic information than just the one floating-point loss that comes out at the end.
For such cases, the fmin function is written to handle dictionary return values.
The idea is that your loss function can return a nested dictionary with all the statistics and diagnostics you want.
The reality is a little less flexible than that though: when using mongodb for example,
the dictionary must be a valid JSON document.
Still, there is lots of flexibility to store domain specific auxiliary results.

When the objective function returns a dictionary, the fmin function looks for some special key-value pairs
in the return value, which it passes along to the optimization algorithm.
There are two mandatory key-value pairs:

* `status` - one of the keys from `hyperopt.STATUS_STRINGS`, such as 'ok' for
  successful completion, and 'fail' in cases where the function turned out to
  be undefined.
* `loss` - the float-valued function value that you are trying to minimize, if
  the status is 'ok' then this has to be present.

The fmin function responds to some optional keys too:

* `attachments` -  a dictionary of key-value pairs whose keys are short
  strings (like filenames) and whose values are potentially long strings (like
  file contents) that should not be loaded from a database every time we
  access the record. (Also, MongoDB limits the length of normal key-value
  pairs so once your value is in the megabytes, you may *have* to make it an
  attachment.)
* `loss_variance` - float - the uncertainty in a stochastic objective function
* `true_loss` - float -
  When doing hyper-parameter optimization, if you store the generalization error of your model with this name, then you can sometimes get spiffier output from the built-in plotting routines.
* `true_loss_variance` - float - the uncertainty in the generalization error

Since dictionary is meant to go with a variety of back-end storage
mechanisms, you should make sure that it is JSON-compatible.  As long as it's
a tree-structured graph of dictionaries, lists, tuples, numbers, strings, and
date-times, you'll be fine.

**HINT:** To store numpy arrays, serialize them to a string, and consider storing
them as attachments.

Writing the function above in dictionary-returning style, it
would look like this:

```python
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK

def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK }

best = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)

print(best)
```

## The Trials Object

To really see the purpose of returning a dictionary,
let's modify the objective function to return some more things,
and pass an explicit `trials` argument to `fmin`.

```python
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(x):
    return {
        'loss': x ** 2,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
        }
trials = Trials()
best = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print(best)
```

In this case the call to fmin proceeds as before, but by passing in a trials object directly,
we can inspect all of the return values that were calculated during the experiment.

So for example:

* `trials.trials` - a list of dictionaries representing everything about the search
* `trials.results` - a list of dictionaries returned by 'objective' during the search
* `trials.losses()` - a list of losses (float for each 'ok' trial)
* `trials.statuses()` - a list of status strings

This trials object can be saved, passed on to the built-in plotting routines,
or analyzed with your own custom code.
Here is a simple example of one way to save and subsequently load a trials object.

```python
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK }

# Initialize an empty trials database
trials = Trials()

# Perform 100 evaluations on the search space
best = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    trials=trials,
    max_evals=100)

# The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method
pickle.dump(trials, open("my_trials.pkl", "wb"))
trials = pickle.load(open("my_trials.pkl", "rb"))

# Perform an additional 100 evaluations
# Note that max_evals is set to 200 because 100 entries already exist in the database
best = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    trials=trials,
    max_evals=200)

print(best)
```

The *attachments* are handled by a special mechanism that makes it possible to use the same code
for both `Trials` and `MongoTrials`.

You can retrieve a trial attachment like this, which retrieves the 'time_module' attachment of the 5th trial:

```python
msg = trials.trial_attachments(trials.trials[5])['time_module']
time_module = pickle.loads(msg)
```

The syntax is somewhat involved because the idea is that attachments are large strings,
so when using MongoTrials, we do not want to download more than necessary.
Strings can also be attached globally to the entire trials object via trials.attachments,
which behaves like a string-to-string dictionary.

**N.B.** Currently, the trial-specific attachments to a Trials object are tossed into the same global trials attachment dictionary, but that may change in the future and it is not true of MongoTrials.

## The `Ctrl` Object for Realtime Communication with MongoDB

It is possible for `fmin()` to give your objective function a handle to the mongodb used by a parallel experiment. This mechanism makes it possible to update the database with partial results, and to communicate with other concurrent processes that are evaluating different points.
Your objective function can even add new search points, just like `rand.suggest`.

The basic technique involves:

* Using the `fmin_pass_expr_memo_ctrl` decorator
* call `pyll.rec_eval` in your own function to build the search space point
  from `expr` and `memo`.
* use `ctrl`, an instance of `hyperopt.Ctrl` to communicate with the live
  trials object.

It's normal if this doesn't make a lot of sense to you after this short tutorial,
but I wanted to give some mention of what's possible with the current code base,
and provide some terms to grep for in the hyperopt source, the unit test,
and example projects, such as [hyperopt-convnet](https://github.com/hyperopt/hyperopt-convnet).
Email me or file a github issue if you'd like some help getting up to speed with this part of the code.
