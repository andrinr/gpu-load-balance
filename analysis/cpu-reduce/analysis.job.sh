#!/bin/bash -l
#SBATCH --job-name="cpuReduce-2" 
#SBATCH --account=uzg2
#SBATCH --constraint=gpu
#SBATCH --time=0-0:01:00
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --ntasks-per-core=1

nThreads=4
p=27
#srun ./a.out $p $nThreads >>results
srun ~dpotter/printaffinity
srun ./a.out $p 1 >>results
srun ./a.out $p 2 >>results
srun ./a.out $p 4 >>results
srun ./a.out $p 8 >>results
srun ./a.out $p 12 >>results
