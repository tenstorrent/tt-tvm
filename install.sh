#!/usr/bin/env bash
set -e # exit if a command fails

if [ -n "$CI_PROJECT_DIR" ]; then
  if [ -z $FORGE_ROOT ]; then export FORGE_ROOT=$CI_PROJECT_DIR; fi
else
  if [ -z $FORGE_ROOT ]; then export FORGE_ROOT=`git rev-parse --show-toplevel`; fi
fi

echo $FORGE_ROOT
cd $FORGE_ROOT
export TVM_HOME=$FORGE_ROOT/third_party/tvm

# Download / untar LLVM
cd $FORGE_ROOT/third_party
if [ ! -d "$FORGE_ROOT/third_party/llvm" ]; then
  wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
  LLVM_TAR=$FORGE_ROOT/third_party/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
  LLVM_DIR=$FORGE_ROOT/third_party/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04
  tar -xf $LLVM_TAR && mv $LLVM_DIR $FORGE_ROOT/third_party/llvm && rm -f $LLVM_TAR
fi

cd $TVM_HOME
git submodule init; git submodule update

mkdir -p build
cp $TVM_HOME/cmake/config.cmake $TVM_HOME/build
cd $TVM_HOME/build

# Link the LLVM thats just been downloaded
LLVM_LINK=$FORGE_ROOT/third_party/llvm/bin/llvm-config
sed -i "s#/usr/bin/llvm-config#$LLVM_LINK#g" $TVM_HOME/build/config.cmake

if [[ "$TVM_BUILD_CONFIG" == "debug" ]]; then
  export TVM_BUILD_CONFIG="Debug"
else
  export TVM_BUILD_CONFIG="Release"
fi

echo $TVM_BUILD_CONFIG
cmake -DCMAKE_BUILD_TYPE=$TVM_BUILD_CONFIG $TVM_HOME
make -j$(nproc)

echo "TVM Built Successful"
cd $FORGE_ROOT

pip install -e third_party/tvm/python

for dir in cutlass cutlass_fpA_intB_gemm/cutlass libflash_attn/cutlass; do
    pushd $TVM_HOME/3rdparty/$dir
    git restore docs media
    popd
done
echo "Files removed during tvm build have been restored."
