# Running unit tests

To run the unit tests, run the `run-tests.sh` script.  You will need to set these environment variables:

- `SPARK_HOME`: your local copy of Apache Spark. Look at `.travis.yml` and `download_spark_dependencies.sh` for details on how to download Apache Spark.
- `HYPEROPT_FMIN_SEED`: the random seed. You need to get its value from `.travis.yml`.

For example:

```bash
hyperopt$ HYPEROPT_FMIN_SEED=3 SPARK_HOME=/usr/local/lib/spark-2.4.4-bin-hadoop2.7 ./run_tests.sh
```

To run the unit test for one file, you can add the file name as the parameter, e.g:

```bash
hyperopt$ HYPEROPT_FMIN_SEED=3 SPARK_HOME=/usr/local/lib/spark-2.4.4-bin-hadoop2.7 ./run_tests.sh hyperopt/tests/test_spark.py
```

To run all unit tests except `test_spark.py`, add the `--no-spark` flag, e.g:

```bash
hyperopt$ HYPEROPT_FMIN_SEED=3 ./run_tests.sh --no-spark
```

To run the unit test for one file other than `test_spark.py`, add the file name as the parameter after the `--no-spark` flag, e.g:

```bash
hyperopt$ HYPEROPT_FMIN_SEED=3 ./run_tests.sh --no-spark test_base.py
```
