#!/bin/bash
set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <testcase>"
  echo "Example: $0 rocket"
  exit 1
fi

TESTCASE=$1
BASE_DIR=$(pwd)
FIR_FILE="${BASE_DIR}/${TESTCASE}.fir"

if [ ! -f "$FIR_FILE" ]; then
  echo "[ERROR] FIR file '${FIR_FILE}' not found."
  exit 1
fi

echo "[INFO] Building testcase $TESTCASE using input: $FIR_FILE"

rm -rf build

export NOOP_HOME="$BASE_DIR"

# Run firtool
/home/xuyinan/tools/circt/build/bin/firtool \
  "$FIR_FILE" \
  -output-annotation-file circt.anno.json \
  --disable-annotation-unknown \
  -load-pass-plugin=/home/xuyinan/tools/chisel-panama-standalone-plugin/build/lib/StandalonePlugin.so \
  -high-firrtl-pass-plugin='firrtl.circuit(annotate-line-coverage,extract-cover-point)' \
  --split-verilog \
  -o=./build/rtl
