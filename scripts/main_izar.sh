#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --chdir /home/catteau/internship
#SBATCH --job-name=main
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem 8G
#SBATCH --cpus-per-task 1
#SBATCH --time=1:00:00

# Load all modules
module load gcc
module load python 
module load cuda 
module load cudnn

# Activate the environment
source .venv/bin/activate
cd Meteo-France/scripts

#srun python experiments_main.py
#srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-1:00:00 python -m python train_neural_net
# srun python -m train_linear_MLP
# srun python -m train_GRIN
# srun python -m reconstruct_GRIN
srun python -m hallucination_GRIN