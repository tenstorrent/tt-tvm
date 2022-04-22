#!/usr/bin/env bash
set -e # exit if a command fails

if [ -n "$CI_PROJECT_DIR" ]; then
  if [ -z $PYBUDA_ROOT ]; then export PYBUDA_ROOT=$CI_PROJECT_DIR; fi
else
  if [ -z $PYBUDA_ROOT ]; then export PYBUDA_ROOT=`git rev-parse --show-toplevel`; fi
fi

echo $PYBUDA_ROOT
cd $PYBUDA_ROOT
export TVM_HOME=$PYBUDA_ROOT/third_party/tvm

# Download / untar LLVM
cd $PYBUDA_ROOT/third_party
if [ ! -d "$PYBUDA_ROOT/third_party/llvm" ]; then
  wget -nc https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
  LLVM_TAR=$PYBUDA_ROOT/third_party/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
  LLVM_DIR=$PYBUDA_ROOT/third_party/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04
  tar -xf $LLVM_TAR && mv $LLVM_DIR $PYBUDA_ROOT/third_party/llvm && rm -f $LLVM_TAR
fi

cd $TVM_HOME
git submodule init; git submodule update

mkdir -p build
cp $TVM_HOME/cmake/config.cmake $TVM_HOME/build
cd $TVM_HOME/build

# Link the LLVM thats just been downloaded
LLVM_LINK=$PYBUDA_ROOT/third_party/llvm/bin/llvm-config
sed -i "s#/usr/bin/llvm-config#$LLVM_LINK#g" $TVM_HOME/build/config.cmake

if [ "$CONFIG" == "debug" ]; then
  export TVM_BUILD_CONFIG="Debug"
else
  export TVM_BUILD_CONFIG="Release"
fi

echo $TVM_BUILD_CONFIG
cmake -j$(nproc) -DCMAKE_BUILD_TYPE=$TVM_BUILD_CONFIG $TVM_HOME
make -j$(nproc)

echo "TVM Built Successful"
cd $PYBUDA_ROOT
