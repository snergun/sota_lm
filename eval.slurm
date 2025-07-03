#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --job-name=ensemble
#SBATCH -o log/slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_gpu  # ... or any appropriate modules
echo "job is starting on `hostname`"
srun python main.py --models AdaptiveInputs KNNLM_v2 MEGA my_adt


