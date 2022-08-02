#!/bin/bash -l
#SBATCH --job-name="cpuReduce-7" 
#SBATCH --account=uzg2
#SBATCH --constraint=gpu
#SBATCH --time=0-0:01:00
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --ntasks-per-core=1

nThreads=128
p=27
srun ./a.out $p $nThreads >>results
