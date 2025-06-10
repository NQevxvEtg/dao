#!/bin/bash

rm -fr build
rm -f chat model.bin
rm -f cmake_configure.txt cmake_build.txt
cmake -S . -B build 2>&1 | tee cmake_configure.txt && cmake --build build 2>&1 | tee cmake_build.txt
mv ./build/chat .
mv ./build/train_tokenizer .
