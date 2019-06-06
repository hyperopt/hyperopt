#!/usr/bin/env bash

# Return on any failure
set -e

# if (got > 1 argument OR ( got 1 argument AND that argument does not exist)) then
# print usage and exit.
if [[ $# -gt 1 || ($# = 1 && ! -e $1) ]]; then
echo "run_tests.sh [target]"
echo ""
echo "Run python tests for this package."
echo "  target -- either a test file or directory [default tests]"
if [[ ($# = 1 && ! -e $1) ]]; then
echo
echo "ERROR: Could not find $1"
fi
exit 1
fi

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
python_major=$($PYSPARK_PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:1])))')

echo $PYSPARK_PYTHON

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
LIBS=$LIBS:$lib
done

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:hyperopt

# This will be used when starting pyspark.
export PYSPARK_SUBMIT_ARGS="--driver-memory 4g --executor-memory 4g pyspark-shell"

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
echo "============= Running the tests in: $noseOptions ============="
$PYSPARK_DRIVER_PYTHON \
-m "nose" \
--nologcapture \
-v --exe "$noseOptions"

done

# nosetests
