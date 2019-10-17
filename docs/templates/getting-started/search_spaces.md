# Defining a Search Space

A search space consists of nested function expressions, including stochastic expressions.
The stochastic expressions are the hyperparameters.
Sampling from this nested stochastic program defines the random search algorithm.
The hyperparameter optimization algorithms work by replacing normal "sampling" logic with
adaptive exploration strategies, which make no attempt to actually sample from the distributions specified in the search space.

It's best to think of search spaces as stochastic argument-sampling programs. For example

```python
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])
```

The result of running this code fragment is a variable `space` that refers to a graph of expression identifiers and their arguments.
Nothing has actually been sampled, it's just a graph describing *how* to sample a point.
The code for dealing with this sort of expression graph is in `hyperopt.pyll` and I will refer to these graphs as *pyll graphs* or *pyll programs*.

If you like, you can evaluate a sample space by sampling from it.

```python
import hyperopt.pyll.stochastic
print hyperopt.pyll.stochastic.sample(space)
```

This search space described by `space` has 3 parameters:

* 'a' - selects the case
* 'c1' - a positive-valued parameter that is used in 'case 1'
* 'c2' - a bounded real-valued parameter that is used in 'case 2'

One thing to notice here is that every optimizable stochastic expression has a *label* as the first argument.
These labels are used to return parameter choices to the caller, and in various ways internally as well.

A second thing to notice is that we used tuples in the middle of the graph (around each of 'case 1' and 'case 2').
Lists, dictionaries, and tuples are all upgraded to "deterministic function expressions" so that they can be part of the search space stochastic program.

A third thing to notice is the numeric expression `1 + hp.lognormal('c1', 0, 1)`, that is embedded into the description of the search space.
As far as the optimization algorithms are concerned, there is no difference between adding the 1 directly in the search space
and adding the 1 within the logic of the objective function itself.
As the designer, you can choose where to put this sort of processing to achieve the kind modularity you want.
Note that the intermediate expression results within the search space can be arbitrary Python objects, even when optimizing in parallel using mongodb.
It is easy to add new types of non-stochastic expressions to a search space description, see below (Section 2.3) for how to do it.

A fourth thing to note is that 'c1' and 'c2' are examples what we will call *conditional parameters*.
Each of 'c1' and 'c2' only figures in the returned sample for a particular value of 'a'.
If 'a' is 0, then 'c1' is used but not 'c2'.
If 'a' is 1, then 'c2' is used but not 'c1'.
Whenever it makes sense to do so, you should encode parameters as conditional ones this way,
rather than simply ignoring parameters in the objective function.
If you expose the fact that 'c1' sometimes has no effect on the objective function (because it has no effect on the argument to the objective function) then search can be more efficient about credit assignment.

## Parameter Expressions

The stochastic expressions currently recognized by hyperopt's optimization algorithms are:

* `hp.choice(label, options)`
  * Returns one of the options, which should be a list or tuple.
       The elements of `options` can themselves be [nested] stochastic expressions.
       In this case, the stochastic choices that only appear in some of the options become *conditional* parameters.

* `hp.randint(label, upper)`
  * Returns a random integer in the range [0, upper). The semantics of this
       distribution is that there is *no* more correlation in the loss function between nearby integer values,
       as compared with more distant integer values.  This is an appropriate distribution for describing random seeds    for example.
       If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either `quniform`, `qloguniform`, `qnormal` or `qlognormal`.

* `hp.uniform(label, low, high)`
  * Returns a value uniformly between `low` and `high`.
  * When optimizing, this variable is constrained to a two-sided interval.

* `hp.quniform(label, low, high, q)`
  * Returns a value like round(uniform(low, high) / q) * q
  * Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below.

* `hp.loguniform(label, low, high)`
  * Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed.
  * When optimizing, this variable is constrained to the interval [exp(low), exp(high)].

* `hp.qloguniform(label, low, high, q)`
  * Returns a value like round(exp(uniform(low, high)) / q) * q
  * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.

* `hp.normal(label, mu, sigma)`
  * Returns a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

* `hp.qnormal(label, mu, sigma, q)`
  * Returns a value like round(normal(mu, sigma) / q) * q
  * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

* `hp.lognormal(label, mu, sigma)`
  * Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed.
        When optimizing, this variable is constrained to be positive.

* `hp.qlognormal(label, mu, sigma, q)`
  * Returns a value like round(exp(normal(mu, sigma)) / q) * q
  * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.

## A Search Space Example: scikit-learn

To see all these possibilities in action, let's look at how one might go about describing the space of hyperparameters of classification algorithms in scikit-learn.
(This idea is being developed in [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn))

```python
from hyperopt import hp
space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
    },
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
            [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    ])
```

## Adding Non-Stochastic Expressions with pyll

You can use such nodes as arguments to pyll functions (see pyll).
File a github issue if you want to know more about this.

In a nutshell, you just have to decorate a top-level (i.e. pickle-friendly) function so
that it can be used via the `scope` object.

```python
import hyperopt.pyll
from hyperopt.pyll import scope

@scope.define
def foo(a, b=0):
     print 'runing foo', a, b
     return a + b / 2

# -- this will print 0, foo is called as usual.
print foo(0)

# In describing search spaces you can use `foo` as you
# would in normal Python. These two calls will not actually call foo,
# they just record that foo should be called to evaluate the graph.

space1 = scope.foo(hp.uniform('a', 0, 10))
space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1))

# -- this will print an pyll.Apply node
print space1

# -- this will draw a sample by running foo()
print hyperopt.pyll.stochastic.sample(space1)
```

## Adding New Kinds of Hyperparameter

Adding new kinds of stochastic expressions for describing parameter search spaces should be avoided if possible.
In order for all search algorithms to work on all spaces, the search algorithms must agree on the kinds of hyperparameter that describe the space.
As the maintainer of the library, I am open to the possibility that some kinds of expressions should be added from time to time, but like I said, I would like to avoid it as much as possible.
Adding new kinds of stochastic expressions is not one of the ways hyperopt is meant to be extensible.
