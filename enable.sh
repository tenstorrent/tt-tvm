#!/usr/bin/env bash

if [ -n "$CI_PROJECT_DIR" ]; then
  if [ -z $ROOT ]; then export ROOT=$CI_PROJECT_DIR; fi
else
  if [ -z $ROOT ]; then export ROOT=`git rev-parse --show-toplevel`; fi
fi

echo $ROOT
export TVM_HOME=$ROOT/third_party/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

