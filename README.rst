hyperopt: Distributed Asynchronous Hyper-parameter Optimization
===============================================================

hyperopt is a Python library for serial and parallel optimization over awkward
search spaces, which may include real-valued, discrete, and conditional
dimensions.

Parallel evaluation is supported by interprocess communication via MongoDB.

Currently only two algorithms are supported:

* Random Search
* TPE

but hyperopt has been designed to accommodate bayesian optimization algorithms
based on e.g. gaussian processes and regression trees, but these are not
currently implemented.


# Installation

Check out the master version of hyperopt from github by typing e.g.

    git clone https://github.com/jaberg/hyperopt.git

Then install it by typing e.g.

    (cd hyperopt && python setup.py install)


# Testing

Run the test suite with `py.test` or `nose`. I use nose myself:

    (cd hyperopt && nosetests)


# Documentation

Tutorial-style documentation is available via

    http://jaberg.github.com/hyperopt


# Examples

    https://github.com/jaberg/hyperopt/wiki/Hyperopt-in-Other-Projects

