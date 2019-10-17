# Scaling out search with Apache Spark

With the new class `SparkTrials`, you can tell Hyperopt to distribute a tuning job across a Spark cluster. Initially developed within Databricks, this API has now been contributed to Hyperopt.

Hyperparameter tuning and model selection often involve training hundreds or thousands of models.  `SparkTrials` runs batches of these training tasks in parallel, one on each Spark executor, allowing massive scale-out for tuning.  To use `SparkTrials` with Hyperopt, simply pass the `SparkTrials` object to Hyperopt’s `fmin()` function:

```python
import hyperopt

best_hyperparameters = hyperopt.fmin(
  fn = training_function,
  space = search_space,
  algo = hyperopt.tpe,
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

## Using SparkTrials in practice

`SparkTrials` takes 2 key parameters: `parallelism` (Maximum number of parallel trials to run, defaulting to the number of Spark executors) and `timeout` (Maximum time in seconds which fmin is allowed to take, defaulting to None).  Timeout provides a budgeting mechanism, allowing a cap on how long tuning can take.

The `parallelism` parameter can be set in conjunction with the `max_evals` parameter in `fmin()`. Hyperopt will test `max_evals` total settings for your hyperparameters, in batches of size `parallelism`.  If `parallelism = max_evals`, then Hyperopt will do Random Search: it will select all hyperparameter settings to test independently and then evaluate them in parallel.  If `parallelism = 1`, then Hyperopt can make full use of adaptive algorithms like Tree of Parzen Estimators (TPE) which iteratively explore the hyperparameter space: each new hyperparameter setting tested will be chosen based on previous results.  Setting `parallelism` in between `1` and `max_evals` allows you to trade off scalability (getting results faster) and adaptiveness (sometimes getting better models).
