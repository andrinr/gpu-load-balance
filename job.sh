
#!/bin/bash -l
#SBATCH --job-name=ORB
#SBATCH --account=uzg2 --constraint=gpu
#SBATCH --time=0-0:30:00 —partition=debug
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --ntasks-per-core=1

srun debug/orbit
