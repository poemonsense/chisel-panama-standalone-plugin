set -e

CIRCT=/home/xuyinan/tools/circt

rm -rf build

mkdir -p build

cd build

cmake .. -GNinja -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCIRCT_DIR=${CIRCT}/build/lib/cmake/circt \
  -DMLIR_DIR=${CIRCT}/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=${CIRCT}/llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ZSTD=Off \
  -DCMAKE_BUILD_TYPE=Release
