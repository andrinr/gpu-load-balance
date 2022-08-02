# Orthogonal Recursive Bisection on the GPU for Accelerated Load Balancing in Large N-Body Simulations

Bachelor Thesis of Andrin Rehmann @UZH 2022. Work in progress. 

Supervised by Douglas Potter and Micheal Böhlen.

Read the [/documentation](https://github.com/andrinr/gpu-load-balance/tree/main/documentation).

## Get started

1. Clone this repo and ``cd`` into its root directory
2. Link mdl2 and blitz from the PKDGRAV repo using ``ln -s /path/to/mdl2`` ``ln -s /path/to/blitz``

### Compile
1. ``mkdir release && cd release``
3. ``cmake ..``
4. `` cmake --build .``

### Run
``./orbit``

### Compile for debugging
1. ``mkdir debug && cd debug``
2. ``cmake -DCMAKE_BUILD_TYPE=Debug --DBZ_DEBUG ..``
3. `` cmake --build .``


### Debug
``gdb ./orbit``
or
``cuda-gdb ./orbit``


## Automized performance analysis

todo

## Open questions

-