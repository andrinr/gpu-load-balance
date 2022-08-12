# Orthogonal Recursive Bisection on the GPU for Accelerated Load Balancing in Large N-Body Simulations

Bachelor Thesis of Andrin Rehmann @UZH 2022. Work in progress. 

Supervised by Douglas Potter and Micheal Böhlen.

Read the [/documentation](https://github.com/andrinr/gpu-load-balance/tree/main/documentation).

## Get started

1. Clone https://bitbucket.org/dpotter/pkdgrav3/
2. Clone this repo and ``cd`` into its root directory
3. Link mdl2 and blitz from the PKDGRAV repo using ``ln -s /path/to/mdl2`` ``ln -s /path/to/blitz``

### Compile
1. ``mkdir release && cd release``
3. ``cmake ..``
4. `` cmake --build .``

### Run
``./orbit`` <x> <y> <o>
where x corresponds to the number of particles 2^x and y to the number of leaf cells in the final tree datastrucutre 2^d. o defines a gpu optimization level where 0 or empty is a cpu only version, 1 corresponds to gpu accelerated version where the bisection method is implented using kernels. Finally 2 further adds a GPU accelerated partitioning method, however this is is still in experimental stages and some bugs are still present.


### Compile for debugging
1. ``mkdir debug && cd debug``
2. ``cmake -DCMAKE_BUILD_TYPE=Debug --DBZ_DEBUG ..``
3. `` cmake --build .``


### Debug
``gdb ./orbit``
or
``cuda-gdb ./orbit``


## Automized performance analysis

For slurm process manager run ``analysis/analysis.sh`` for a test with variable particle counts and ``analysis/analysis2.sh`` for a test with variable domain counts. 
