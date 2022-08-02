#!/bin/bash -l
#SBATCH --job-name="cpuReduce-7" 
#SBATCH --account=uzg2
#SBATCH --constraint=gpu
#SBATCH --time=0-0:01:00
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --ntasks-per-core=1

pParticles=128
pDomains=27
opt=0
outFile="results0"
srun ../build/orbit $pParticles $pDomains $opt >>$outFile
