#!/usr/bin/env bash

cd eaconv/src
echo "Compiling Efficient Aggregation Convolution kernels by nvcc..."
rm -f EAConv2d_kernel.o
rm -rf ../_ex

cd ../..
nvcc -c -o eaconv/src/EAConv2d_kernel.o eaconv/src/EAConv2d_kernel.cu -x cu -Xcompiler -fPIC -std=c++11 \
-I ~/anaconda3/lib/python3.6/site-packages/torch/lib/include/ \
-I ~/anaconda3/lib/python3.6/site-packages/torch/lib/include/TH \
-I ~/anaconda3/lib/python3.6/site-packages/torch/lib/include/THC
nvcc -c -o eaconv/src/handle.o eaconv/src/handle.cu -x cu -Xcompiler -fPIC -std=c++11
nvcc -c -o eaconv/src/conv_params.o eaconv/src/conv_params.cu -x cu -Xcompiler -fPIC -std=c++11
cd eaconv/
python3 build.py
