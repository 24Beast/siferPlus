#!/bin/bash

#SBATCH -G a30:1
#SBATCH -c 32
#SBATCH --mem 160G
#SBATCH -p general
#SBATCH -t 2-00:00:00   # time in d-hh:mm:ss



module purge

module load mamba/latest

source activate SiferPlus

python3 main.py