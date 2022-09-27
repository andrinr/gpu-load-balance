# Orthogonal Recursive Bisection on the GPU for Accelerated Load Balancing in Large N-Body Simulations

Bachelor Thesis of Andrin Rehmann @UZH 2022.

## Abstract 

Large simulations with billions of particles are used to understand the universe.
Commonly, the fast multipole method is employed to improve the runtime,
where a space partitioning binary tree data structure is required. The structure is generated using the Orthogonal Recursive Bisection method and at the
same time used as a load balancing strategy to leverage the hardware of super
computers optimally.
In this paper, a GPU accelerated version of ORB is proposed and implemented in the CUDA programming language. The implementation manages to maintain a consistent runtime during the entire execution of the ORB
method, regardless of the increasing complexity as fragmentation grows with
the tree depth and the number of leaf cells. Performance measurements showed
a speedup by a factor of 5.4 over a fully parallelized and optimized CPU version.

Read the entire [/thesis](https://github.com/andrinr/gpu-load-balance/tree/main/andrin_rehmann_bsc_thesis.pdf).

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
where x corresponds to the number of particles 2^x and y to the number of leaf cells in the final tree datastrucutre 2^y. o defines a gpu optimization level where 0 or empty is a cpu only version, 1 corresponds to gpu accelerated version where the bisection method is implented using kernels. Finally 2 further adds a GPU accelerated partitioning method, however this is is still in experimental stages and some bugs are still present.


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
