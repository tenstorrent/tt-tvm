#!/usr/bin/env bash

if [ -n "$CI_PROJECT_DIR" ]; then
  if [ -z $PYBUDA_ROOT ]; then export PYBUDA_ROOT=$CI_PROJECT_DIR; fi
else
  if [ -z $PYBUDA_ROOT ]; then export PYBUDA_ROOT=`git rev-parse --show-toplevel`; fi
fi

echo $PYBUDA_ROOT
export TVM_HOME=$PYBUDA_ROOT/third_party/tvm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVM_HOME/build
pip install -e third_party/tvm/python

cd $PYBUDA_ROOT