# #!/bin/bash
# #SBATCH --output=output.out
# #SBATCH --error=error.err
# #SBATCH --chdir /home/catteau/internship
# #SBATCH --job-name=main
# #SBATCH --partition gpu
# #SBATCH --gres gpu:1
# #SBATCH --mem 8G
# #SBATCH --cpus-per-task 1
# #SBATCH --time=0:30:00

# # Load all modules
# module load gcc
# module load python 
# module load cuda 
# module load cudnn

# # Activate the environment
# source .venv/bin/activate
# cd Meteo-France/scripts

python3 -m train_diffusion_MLP
# python3 -m train_MLP

for mask_proba in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "Running with masking probability: $mask_proba"
    # srun python -m train_MLP
    python3 -m reconstruct_MLP $mask_proba
done