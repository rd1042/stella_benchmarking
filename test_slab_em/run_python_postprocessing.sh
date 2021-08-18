#!/bin/bash
#SBATCH -N 1
#SBATCH -A FUA35_STELTURB
#SBATCH -p skl_fua_prod
#SBATCH --time 00:10:00
#SBATCH --job-name=python
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bob.davies@york.ac.uk

NUMPROC=1

loadpy
python save_phi_vs_t.py
