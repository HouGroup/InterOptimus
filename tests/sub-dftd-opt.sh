#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 96
source /public5/soft/modules/module.sh
module load mpi/oneAPI/2021.2
export PATH=/public5/home/t6s001944/software-t6s001944/vasp.6.3.0/bin:$PATH
export LD_LIBRARY_PATH=/public5/home/t6s001944/software-t6s001944/vasp.6.3.0/dftd4/install/lib64:$LD_LIBRARY_PATH
mpirun -np 96 vasp_std
