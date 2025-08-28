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

cd $TVM_HOME
git submodule init; git submodule update

mkdir -p build
cp $TVM_HOME/cmake/config.cmake $TVM_HOME/build
cd $TVM_HOME/build

if [[ -n $LLVM_CONFIG_CMD && -e $LLVM_CONFIG_CMD ]]; then
    # pass in through environment variable
    echo Using "$LLVM_CONFIG_CMD" as LLVM_LINK
    LLVM_LINK="$LLVM_CONFIG_CMD"
elif [[ -e /usr/bin/llvm-config-17 ]]; then
    # should be present; llvm-17 is included in our ubuntu 22.04 images
    echo Using /usr/bin/llvm-config-17 as LLVM_LINK
    LLVM_LINK=/usr/bin/llvm-config-17
else
    # Download / untar LLVM
    echo Downloading llvm 13
    if [ ! -d "$PYBUDA_ROOT/third_party/llvm" ]; then
        wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
        LLVM_TAR=$PYBUDA_ROOT/third_party/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
        LLVM_DIR=$PYBUDA_ROOT/third_party/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04
        tar -xf $LLVM_TAR && mv $LLVM_DIR $PYBUDA_ROOT/third_party/llvm && rm -f $LLVM_TAR
    fi
    # Link the LLVM thats just been downloaded
    LLVM_LINK=$PYBUDA_ROOT/third_party/llvm/bin/llvm-config
fi
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
