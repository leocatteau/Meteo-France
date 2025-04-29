#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --chdir /home/catteau/internship
#SBATCH --job-name=main
#SBATCH --ntasks=30
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=1:00:00

# Load all modules
module load gcc
module load python 
module load openmpi
module load py-mpi4py

# Activate the environment
cd ..
cd ..
source .venv/bin/activate
cd Meteo-France/scripts

#srun python experiments_main.py
srun --partition gpu --qos gpu --gres gpu:1 --mem 8G --cpus-per-task 5 --time 0-2:00:00 python -m python train_neural_net