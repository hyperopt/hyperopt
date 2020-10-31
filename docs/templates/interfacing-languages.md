# Interfacing Hyperopt with other programming languages

There are basically two ways to interface hyperopt with other languages:

1. you can write a Python wrapper around your cost function that is not written in Python, or
2. you can replace the `hyperopt-mongo-worker` program and communicate with MongoDB directly using JSON.

## Wrapping a call to non-Python code

The easiest way to use hyperopt to optimize the arguments to a non-python function, such as for example an external executable, is to write a Python function wrapper around that external executable. Supposing you have an executable `foo` that takes an integer command-line argument `--n` and prints out a score, you might wrap it like this:

```python
import subprocess
def foo_wrapper(n):
    # Optional: write out a script for the external executable
    # (we just call foo with the argument proposed by hyperopt)
    proc = subprocess.Popen(['foo', '--n', n], stdout=subprocess.PIPE)
    proc_out, proc_err = proc.communicate()
    # <you might have to do some more elaborate parsing of foo's output here>
    score = float(proc_out)
    return score
```

Of course, to optimize the `n` argument to `foo` you also need to call hyperopt.fmin, and define the search space. I can only imagine that you will want to do this part in Python.

```python
from hyperopt import fmin, hp, rand

best_n = fmin(foo_wrapper, hp.quniform('n', 1, 100, 1), algo=rand.suggest)

print best_n
```

When the search space is larger than the simple one here, you might want or need the wrapper function to translate its argument into some kind of configuration file/script for the external executable.

This approach is perfectly compatible with MongoTrials.

## Communicating with MongoDB Directly

It is possible to interface more directly with the search process (when using MongoTrials) by communicating with MongoDB directly, just like `hyperopt-mongo-worker` does. It's beyond the scope of a tutorial to explain how to do this, but Hannes Schultz (@temporaer) got hyperopt working with his MDBQ project, which is a standalone mongodb-based task queue:

[Hyperopt C++ Client](https://github.com/temporaer/MDBQ/blob/master/src/example/hyperopt_client.cpp)

Have a look at that code, as well as the contents of [hyperopt/mongoexp.py](https://github.com/jaberg/hyperopt/blob/master/hyperopt/mongoexp.py) to understand how worker processes are expected to reserve jobs in the work queue, and store results back to MongoDB.
