#!/usr/bin/env bash

echo "Downloading Spark if necessary"
echo "Spark version = $SPARK_VERSION"
echo "Spark build = $SPARK_BUILD"
echo "Spark build URL = $SPARK_BUILD_URL"

sparkVersionsDir="$HOME/.cache/spark-versions"
mkdir -p "$sparkVersionsDir"
SPARK_BUILD_DIR="$sparkVersionsDir/$SPARK_BUILD"

if [[ -d "$SPARK_BUILD_DIR" ]]; then
    echo "Skipping download - found Spark dir $SPARK_BUILD_DIR"
else
    echo "Missing $SPARK_BUILD_DIR, downloading archive"
    filename="$HOME/.cache/spark-versions/$SPARK_BUILD.tgz"

    if ! [[ -d "$SPARK_BUILD_DIR" ]]; then
            echo "Downloading $SPARK_BUILD_URL ..."
            wget "$SPARK_BUILD_URL" -O $filename
            echo "[Debug] Following should list a valid spark binary"
            ls -larth $HOME/.cache/spark-versions/*
            tar -xzf $filename --directory $HOME/.cache/spark-versions > /dev/null
    fi

    echo "Content of $SPARK_BUILD_DIR:"
    ls -la "$SPARK_BUILD_DIR"
fi

