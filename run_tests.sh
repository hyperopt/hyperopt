#!/usr/bin/env bash

use_spark=true
while :; do
case $1 in
--no-spark)
use_spark=false
;;
*) break
esac
shift
done

# Return on any failure
set -e

# if (got > 1 argument OR ( got 1 argument AND that argument does not exist)) then
# print usage and exit.
if [[ $# -gt 1 || ($# = 1 && ! -e $1) ]]; then
echo "run_tests.sh [--no-spark] [target]"
echo ""
echo "Run python tests for this package."
echo "  --no-spark : flag to tell this script to skip Apache Spark-related tests"
echo "  target : either a test file or directory [default: tests]"
if [[ ($# = 1 && ! -e $1) ]]; then
echo
echo "ERROR: Could not find $1"
fi
exit 1
fi

if [[ ("$use_spark" = false) && ($1 == *test_spark.py) ]] ; then
echo
echo "ERROR: Cannot run $1 with --no-spark flag"
exit 1
fi

if [[ "$use_spark" = true ]]; then
# assumes run from the package base directory
if [ -z "$SPARK_HOME" ]; then
echo 'You need to set $SPARK_HOME to run these tests.' >&2
exit 1
fi

# Honor the choice of python driver
if [ -z "$PYSPARK_PYTHON" ]; then
PYSPARK_PYTHON=`which python`
fi
# Override the python driver version as well to make sure we are in sync in the tests.
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON

echo $PYSPARK_PYTHON

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
LIBS=$LIBS:$lib
done

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:hyperopt

# This will be used when starting pyspark.
export PYSPARK_SUBMIT_ARGS="--driver-memory 4g --executor-memory 4g pyspark-shell"
fi

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run test suites

if [ -f "$1" ]; then
noseOptionsArr="$1"
else
if [ -d "$1" ]; then
targetDir=$1
else
targetDir=$DIR/hyperopt/tests
fi
# add all python files in the test dir recursively
echo "============= Searching for tests in: $targetDir ============="
noseOptionsArr="$(find "$targetDir" -type f | grep "\.py" | grep -v "\.pyc" | grep -v "\.py~" | grep -v "__init__.py")"
fi

for noseOptions in $noseOptionsArr
do
if [[ ("$use_spark" = false) && ($noseOptions == *test_spark.py) ]] ; then
continue
fi
echo "============= Running the tests in: $noseOptions ============="
if [[ "$use_spark" = true ]]; then
$PYSPARK_DRIVER_PYTHON \
-m "nose" \
--nologcapture \
-v --exe "$noseOptions"
else
python \
-m "nose" \
--nologcapture \
-v --exe "$noseOptions"
fi
done

# nosetests
