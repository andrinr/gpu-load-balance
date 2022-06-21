# Orthogonal Recursive Bisection on the GPU for Accelerated Load Balancing in Large N-Body Simulations

Bachelor Thesis of Andrin Rehmann

Supervised by Douglas Potter and Micheal Böhlen

Head to [/documentation](https://github.com/andrinr/gpu-load-balance/tree/main/documentation) for the writeup.

## Get started

1. OpenMPI ``sudo apt-get install openmpi-bin libopenmpi-dev``
2. Blitz++ https://github.com/blitzpp/blitz

### Compile
1. ``mkdir build``
2. ``cd build``
3. ``cmake ..`` or for debugging ``cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo --DBZ_DEBUG ..``
4. `` cmake --build .``


### Debug
``mpirun -np <x> gdb glb <# thousand of particles> <# domains>``

Or with   console for each process:

``mpirun -np <x> xterm -e gdb glb <# thousand of particles> <# domains>``

Or run gdb on only one process:

``mpiexec -n 1 gdb glb <# thousand of particles> <# domains>m : -n <x-1> glb <# thousand of particles> <# domains>``

### CUDA Profile

Using Nvidia Nsight: 
```run ./nsys-ui```

### Run
``mpirun -np <x> glb <# thousand of particles> <# domains>``

## Automized performance analysis

Insert instructions here

## Open questions
- How are number of threads defined?
- How can I pass header data?
- Why cant I cast to input data?
- 