# Parallelized ORB with MPI

## Get started

1. OpenMPI ``sudo apt-get install openmpi-bin libopenmpi-dev``
2. Blitz++ https://github.com/blitzpp/blitz

## Compile
``make``

## Debug
``mpirun -np <x> gdb build/final_program``

Or with   console for each process:

``mpirun -np <x> xterm -e gdb build/final_program ``

Or run gdb on only one process:

``mpiexec -n 1 gdb build/final_program : -n <x-1> build/final_program``

## Run
``mpirun -np <x> build/final_program``
