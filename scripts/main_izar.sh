#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --chdir /home/catteau/internship
#SBATCH --job-name=main
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem 8G
#SBATCH --cpus-per-task 1
#SBATCH --time=0:30:00

# Load all modules
module load gcc
module load python 
module load cuda 
module load cudnn

# Activate the environment
source .venv/bin/activate
cd Meteo-France/scripts

# root_path='../../datasets/'

#srun python experiments_main.py
# srun python -m train_linear --root_path $root_path
# srun python -m train_MLP --root_path $root_path
srun python -m train_GRIN --root_path $root_path
# srun python -m reconstruct_GRIN
# srun python -m hallucination_GRIN