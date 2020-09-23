# Getting started with Hyperopt

Hyperopt's job is to find the best value of a scalar-valued, possibly-stochastic function over a set of possible arguments to that function.

Whereas many optimization packages will assume that these inputs are drawn from a vector space,
Hyperopt is different in that it encourages you to describe your search space in more detail.
By providing more information about where your function is defined, and where you think the best values are, you allow algorithms in hyperopt to search more efficiently.

The way to use hyperopt is to describe:

* the objective function to minimize
* the space over which to search
* the database in which to store all the point evaluations of the search
* the search algorithm to use

This (most basic) tutorial will walk through how to write functions and search spaces,
using the default `Trials` database, and the dummy `rand` (random) search algorithm.
Section (1) is about the different calling conventions for communication between an objective function and hyperopt.
Section (2) is about describing search spaces.

Parallel search is possible when replacing the `Trials` database with
a `MongoTrials` one;
there is another wiki page on the subject of [using mongodb for parallel search](Parallelizing-Evaluations-During-Search-via-MongoDB).

Choosing the search algorithm is as simple as passing `algo=hyperopt.tpe.suggest` instead of `algo=hyperopt.rand.suggest`.
The search algorithms are actually callable objects, whose constructors
accept configuration arguments, but that's about all there is to say about the
mechanics of choosing a search algorithm.
