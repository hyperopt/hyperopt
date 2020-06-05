
# Hyperopt: Distributed Hyperparameter Optimization

<p align="center">
<img src="https://i.postimg.cc/TPmffWrp/hyperopt-new.png" />
</p>


[![Build Status](https://travis-ci.org/hyperopt/hyperopt.svg?branch=master)](https://travis-ci.org/hyperopt/hyperopt)  [![PyPI version](https://badge.fury.io/py/hyperopt.svg)](https://badge.fury.io/py/hyperopt)  [![Anaconda-Server Badge](https://anaconda.org/conda-forge/hyperopt/badges/version.svg)](https://anaconda.org/conda-forge/hyperopt)

[Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for serial and parallel optimization over awkward
search spaces, which may include real-valued, discrete, and conditional
dimensions.

## Getting started

Install hyperopt from PyPI

```bash
$ pip install hyperopt
```

to run your first example

```python
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}
```

## Contributing 

### Setup
If you're a developer, clone this repository and install from source:

```bash
$ git clone https://github.com/hyperopt/hyperopt.git
$ cd hyperopt

# Create a virtual env (python 3.x) to contain the dependencies and activate it
$ python3 -m venv venv
$ source venv/bin/activate

$ python setup.py develop &&  pip install -e '.[MongoTrials, SparkTrials, ATPE, dev]'
```

Note that dev dependencies require python 3.6+.

### Running tests
The tests for this project use [PyTest](https://docs.pytest.org/en/latest/) and can be run by calling `pytest`.

### Formatting 
We recommend to use [Black](https://github.com/psf/black) to format your code before submitting a PR. You can use it 
with a pre-commit hook as follows:

```bash
$ pip install pre-commit
$ pre-commit install
```

Then, once you commit ensure that git hooks are activated (Pycharm for example has the 
option to omit them). This will run black automatically on all files you modified, 
failing if there are any files requiring to be blacked.

```bash
$ black {source_file_or_directory}
```

## Algorithms

Currently three algorithms are implemented in hyperopt:

- [Random Search](http://www.jmlr.org/papers/v13/bergstra12a.html?source=post_page---------------------------)
- [Tree of Parzen Estimators (TPE)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- [Adaptive TPE](https://www.electricbrain.io/blog/learning-to-optimize)

Hyperopt has been designed to accommodate Bayesian optimization algorithms based on Gaussian processes and regression trees, but these are not currently implemented.

All algorithms can be parallelized in two ways, using:

- [Apache Spark](https://spark.apache.org/)
- [MongoDB](https://mongodb.com)

## Documentation

[Hyperopt documentation can be found here](http://hyperopt.github.io/hyperopt), but is partly still hosted on the wiki. Here are some quick links to the most relevant pages:

- [Basic tutorial](https://github.com/hyperopt/hyperopt/wiki/FMin)
- [Installation notes](https://github.com/hyperopt/hyperopt/wiki/Installation-Notes)
- [Using mongodb](https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB)

## Related Projects

* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
* [hyperopt-nnet](https://github.com/hyperopt/hyperopt-nnet)
* [hyperas](https://github.com/maxpumperla/hyperas)
* [hyperopt-convent](https://github.com/hyperopt/hyperopt-convnet)
* [hyperopt-gpsmbo](https://github.com/hyperopt/hyperopt-gpsmbo/blob/master/hp_gpsmbo/hpsuggest.py)

## Examples

See [projects using hyperopt](https://github.com/hyperopt/hyperopt/wiki/Hyperopt-in-Other-Projects) on the wiki.

## Announcements mailing list

[Announcments](https://groups.google.com/forum/#!forum/hyperopt-announce)

## Discussion mailing list

[Discussion](https://groups.google.com/forum/#!forum/hyperopt-discuss)

## Cite

If you use this software for research, plase cite the following paper:

Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).

## Thanks

This project has received support from

- National Science Foundation (IIS-0963668),
- Banting Postdoctoral Fellowship program,
- National Science and Engineering Research Council of Canada (NSERC),
- D-Wave Systems, Inc.
