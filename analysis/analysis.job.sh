#!/bin/bash -l
#SBATCH --job-name="orbit-1-30" 
#SBATCH --account=uzg2
#SBATCH --constraint=gpu
#SBATCH --time=0-0:01:00
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --ntasks-per-core=1

pParticles=30
pDomains=10
opt=1
outFile="results1"
srun ../build/orbit $pParticles $pDomains $opt >>$outFile
