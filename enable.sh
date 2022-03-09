#!/usr/bin/env bash
set -e # exit if a command fails

echo $ROOT
export TVM_HOME=$ROOT/third_party/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

cd $ROOT
