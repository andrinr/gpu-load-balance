# Orthogonal Recursive Bisection on the GPU for Accelerated Load Balancing in Large N-Body Simulations

Bachelor Thesis of Andrin Rehmann

Supervised by Douglas Potter and Micheal Böhlen

Head to [/documentation](https://github.com/andrinr/gpu-load-balance/tree/main/documentation) for the writeup.

## Get started

1. OpenMPI ``sudo apt-get install openmpi-bin libopenmpi-dev``
2. Blitz++ https://github.com/blitzpp/blitz
3. Clone PKDGRAV3 from https://bitbucket.org/dpotter/pkdgrav3/ and make sure its prerequisites are met.
4. Clone this repo and ``cd`` into its root directory
5. Link mdl2 and blitz from the PKDGRAV repo using ``ln -s /path/to/lib``

### Compile for execution
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

### CUDA Profile

Using Nvidia Nsight: 
```run ./nsys-ui```


## Automized performance analysis

todo

## Open questions
- Should we use uint4 for shared memory?
- Why does warp occupancy erode over time?
- Why even "mitigate" first two threads if iterating over certain number of elements anyways?
- 
- 