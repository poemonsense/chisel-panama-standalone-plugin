#!/bin/bash
set -e

cd build && ninja && cd ..

if [ $# -eq 1 ]; then
  cd tests && bash build.sh "$1" && cd ..
fi
