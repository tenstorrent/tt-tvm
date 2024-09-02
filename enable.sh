#!/usr/bin/env bash

if [ -n "$CI_PROJECT_DIR" ]; then
  if [ -z $FORGE_ROOT ]; then export FORGE_ROOT=$CI_PROJECT_DIR; fi
else
  if [ -z $FORGE_ROOT ]; then export FORGE_ROOT=`git rev-parse --show-toplevel`; fi
fi

export TVM_HOME=$FORGE_ROOT/third_party/tvm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVM_HOME/build

cd $FORGE_ROOT
