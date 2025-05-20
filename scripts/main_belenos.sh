#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --chdir /home/catteau/internship
#SBATCH --job-name=main
#SBATCH --partition=ndl
#SBATCH --mem-per-cpu=8G
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:00

# Load all modules
module load gcc
module load python 
module load cuda 
module load cudnn

# Activate the environment
source .venv/bin/activate
cd Meteo-France/scripts

data_path="/scratch/work/catteaul/datasets/"

#srun python experiments_main.py
srun python -m train_linear $data_path
# srun python -m train_GRIN $data_path
# srun python -m reconstruct_GRIN
# srun python -m hallucination_GRIN