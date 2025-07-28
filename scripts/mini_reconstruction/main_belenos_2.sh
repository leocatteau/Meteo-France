#!/bin/bash
#SBATCH --output=output_processed.out
#SBATCH --error=error_processed.err
#SBATCH --job-name=imputation_processed
#SBATCH --partition=ndl
#SBATCH --time=01:00:00 #9h pour 200 steps de train, 40min pour reconstruction 
#SBATCH --mem=0

# Load all modules
module load gcc
# module load python 
# module load cuda 
# module load cudnn

# Activate the environment
# source .venv/bin/activate
# conda activate .venv
# cd /workdir/Meteo-France/scripts

data_path="/scratch/work/catteaul/datasets/"

#srun python experiments_main.py
# srun python -m train_diffusion_GRIN_2
srun python -m reconstruct_GRIN2_processed