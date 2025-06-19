#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --job-name=imputation
#SBATCH --partition=ndl
#SBATCH --time=0:05:00

# Load all modules
# module load gcc
# module load python 
# module load cuda 
# module load cudnn

# Activate the environment
# source .venv/bin/activate
# conda activate .venv
# cd /workdir/Meteo-France/scripts

data_path="/scratch/work/catteaul/datasets/"

#srun python experiments_main.py
srun python -m train_linear $data_path
# srun python -m train_GRIN $data_path
# srun python -m reconstruct_GRIN
# srun python -m hallucination_GRIN