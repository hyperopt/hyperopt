# Scaling out search with Apache Spark

With the new class `SparkTrials`, you can tell Hyperopt to distribute a tuning job across an Apache Spark cluster. Initially developed within Databricks, this API has now been contributed to Hyperopt.

Hyperparameter tuning and model selection often involve training hundreds or thousands of models.  `SparkTrials` runs batches of these training tasks in parallel, one on each Spark executor, allowing massive scale-out for tuning.  To use `SparkTrials` with Hyperopt, simply pass the `SparkTrials` object to Hyperopt’s `fmin()` function:

```python
import hyperopt

best_hyperparameters = hyperopt.fmin(
  fn = training_function,
  space = search_space,
  algo = hyperopt.tpe.suggest,
  max_evals = 64,
  trials = hyperopt.SparkTrials())
```

Under the hood, `fmin()` will generate new hyperparameter settings to test and pass them to `SparkTrials`, which runs these tasks asynchronously on a cluster as follows:

- Hyperopt’s primary logic runs on the Spark driver, computing new hyperparameter settings.
- When a worker is ready for a new task, Hyperopt kicks off a single-task Spark job for that hyperparameter setting.
- Within that task, which runs on one Spark executor, user code will be executed to train and evaluate a new ML model.
- When done, the Spark task will return the results, including the loss, to the driver.  

These new results are used by Hyperopt to compute better hyperparameter settings for future tasks.

Since `SparkTrials` fits and evaluates each model on one Spark worker, it is limited to tuning single-machine ML models and workflows, such as scikit-learn or single-machine TensorFlow.  For distributed ML algorithms such as Apache Spark MLlib or Horovod, you can use Hyperopt’s default Trials class.

## SparkTrials API

`SparkTrials` may be configured via 3 arguments, all of which are optional:

**`parallelism`**
The maximum number of trials to evaluate concurrently. Greater parallelism allows scale-out testing of more hyperparameter settings. Defaults to Spark `SparkContext.defaultParallelism`.

* Trade-offs: The `parallelism` parameter can be set in conjunction with the `max_evals` parameter in `fmin()`. Hyperopt will test `max_evals` total settings for your hyperparameters, in batches of size `parallelism`.  If `parallelism = max_evals`, then Hyperopt will do Random Search: it will select all hyperparameter settings to test independently and then evaluate them in parallel.  If `parallelism = 1`, then Hyperopt can make full use of adaptive algorithms like Tree of Parzen Estimators (TPE) which iteratively explore the hyperparameter space: each new hyperparameter setting tested will be chosen based on previous results.  Setting `parallelism` in between `1` and `max_evals` allows you to trade off scalability (getting results faster) and adaptiveness (sometimes getting better models).
* Limits: There is currently a hard cap on parallelism of 128. `SparkTrials` will also check the cluster’s configuration to see how many concurrent tasks Spark will allow; if parallelism exceeds this maximum, `SparkTrials` will reduce parallelism to this maximum.

**`timeout`**
Maximum time in seconds which `fmin()` is allowed to take, defaulting to None.
Timeout provides a budgeting mechanism, allowing a cap on how long tuning can take.  When the timeout is hit, runs are terminated if possible, and `fmin()` exits, returning the current set of results.

**`spark_session`**
`SparkSession` instance for `SparkTrials` to use.  If this is not given, `SparkTrials` will look for an existing `SparkSession`.

The `SparkTrials` API may also be viewed by calling `help(SparkTrials)`.


## Example workflow with SparkTrials

Below, we give an example workflow which tunes a scikit-learn model using SparkTrials.	 This example was adapted from the [scikit-learn doc example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html) for sparse logistic regression with MNIST.

```python
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK

# Load MNIST data, and preprocess it by standarizing features.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# First, set up the scikit-learn workflow, wrapped within a function.
def train(params):
  """
  This is our main training function which we pass to Hyperopt.
  It takes in hyperparameter settings, fits a model based on those settings,
  evaluates the model, and returns the loss.
  
  :param params: map specifying the hyperparameter settings to test
  :return: loss for the fitted model
  """
  # We will tune 2 hyperparameters:
  #  regularization and the penalty type (L1 vs L2).
  regParam = float(params['regParam'])
  penalty = params['penalty']

  # Turn up tolerance for faster convergence
  clf = LogisticRegression(C=1.0 / regParam,
                           multi_class='multinomial',
                           penalty=penalty, solver='saga', tol=0.1)
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)

  return {'loss': -score, 'status': STATUS_OK}

# Next, define a search space for Hyperopt.
search_space = {
  'penalty': hp.choice('penalty', ['l1', 'l2']),
  'regParam': hp.loguniform('regParam', -10.0, 0),
}

# Select a search algorithm for Hyperopt to use.
algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

# We can run Hyperopt locally (only on the driver machine)
# by calling `fmin` without an explicit `trials` argument.
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=32)
best_hyperparameters

# We can distribute tuning across our Spark cluster
# by calling `fmin` with a `SparkTrials` instance.
spark_trials = SparkTrials()
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  trials=spark_trials,
  max_evals=32)
best_hyperparameters
```
