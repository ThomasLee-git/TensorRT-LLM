#!/bin/bash

build_root="tmp_build"
# rm -r ${build_root}
# mkdir -p ${build_root}
cmake -G Ninja -B ${build_root} -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build ${build_root}