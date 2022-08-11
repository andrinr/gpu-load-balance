#!/bin/bash -l
#SBATCH --job-name="orbit-1-30" 
#SBATCH --account=uzg2
#SBATCH --constraint=gpu
#SBATCH --time=0-0:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --ntasks-per-core=1

srun a.out >> results
