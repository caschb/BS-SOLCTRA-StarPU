#!/bin/sh
#SBATCH --job-name=bs-solctra-multinode
#SBATCH --partition=nukwa-wide
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

export ID=${SLURM_JOB_ID}
export STARPU_SCHED=dmda
export STARPU_RESERVE_NCPU=1
export STARPU_WORKERS_GET_BIND=0

if [ ! -d "./results_${ID}" ]; then
  mkdir "./results_${ID}"
fi

mpiexec -np 2\
    ../build/main/bs_solctra\
    --total-particles 1024\
    --particles ../data/input_big.txt\
    --job-id $ID\
    --resource-path ../data/resources\
    --steps 1000\
    --mode 1\
    --debug 1\
    --magnetic_prof 0 100 0 2
