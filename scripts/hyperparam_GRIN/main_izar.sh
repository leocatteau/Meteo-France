#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --chdir /home/catteau/internship
#SBATCH --job-name=main
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem 8G
#SBATCH --cpus-per-task 1
#SBATCH --time=2:30:00

# Load all modules
module load gcc
module load python 
module load cuda 
module load cudnn

# Activate the environment
source .venv/bin/activate
cd Meteo-France/scripts/loss

for hidden_dim in 100
do
    echo "Running with hidden dimension: $hidden_dim"
    srun python -m train_GRIN $hidden_dim
done