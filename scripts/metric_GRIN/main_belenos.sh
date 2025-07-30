#!/bin/bash
#SBATCH --output=output_graph_partial.out
#SBATCH --error=error_graph_partial.err
#SBATCH --job-name=train_graph_partial
#SBATCH --partition=ndl
#SBATCH --time=2:00:00

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
srun python -m train_diffusion_GRIN
# srun python -m train_GRIN $data_path
# srun python -m reconstruct_GRIN
# srun python -m hallucination_GRIN