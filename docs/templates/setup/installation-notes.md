# Installation notes for hyperopt

## MongoDB

Hyperopt requires [mongodb](http://www.mongodb.org) (sometimes "mongo" for short) to perform parallel search. As far as I know, hyperopt is compatible with all versions in the 2.x.x series, which is the current one ([download the latest version here](http://www.mongodb.org/downloads)). It might even be compatible with all versions ever of mongodb, I don't know of any particular version requirements on mongo.

On linux and OSX, once you have downloaded mongodb and unpacked it, simply symlink it into the `bin/` subdirectory of your virtualenv and your installation is complete.

```bash
# from the root of your virtualenv
# (or basically any folder with an active bin/ subdirectory)
(cd bin && { for F in ../mongodb-linux-x86_64-2.2.2/bin/* ; do echo "linking $F" ; ln -s $F ; done } )
```

Verify that hyperopt can use mongod by running either the full unit test suite, or just the mongo file

```bash
# cd to the hyperopt project root
nosetests hyperopt/tests/test_mongoexp.py
```

## Spark

We have a little [script](https://github.com/hyperopt/hyperopt/blob/master/download_spark_dependencies.sh) that will
help you download all necessary dependencies.