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
# cp $TVM_HOME/cmake/config.cmake $TVM_HOME/build
cd $TVM_HOME

# Link the LLVM thats just been downloaded
# LLVM_LINK=/Users/aknezevic/work/tt-mlir/./.local/toolchain/bin/llvm-config
# LLVM_LINK=$PYBUDA_ROOT/third_party/llvm/bin/llvm-config
# sed -i"" -e "s#/usr/bin/llvm-config#$LLVM_LINK#g" $TVM_HOME/build/config.cmake

if [[ "$TVM_BUILD_CONFIG" == "debug" ]]; then
  export TVM_BUILD_CONFIG="Debug"
else
  export TVM_BUILD_CONFIG="Release"
fi

echo $TVM_BUILD_CONFIG
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DUSE_LLVM=OFF
cmake --build build

echo "TVM Built Successful"
cd $FORGE_ROOT

pip install -e third_party/tvm/python

for dir in cutlass cutlass_fpA_intB_gemm/cutlass libflash_attn/cutlass; do
    pushd $TVM_HOME/3rdparty/$dir
    git restore docs media
    popd
done
echo "Files removed during tvm build have been restored."
