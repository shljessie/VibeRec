#!/bin/bash
#SBATCH --job-name=blip2
#SBATCH --output=./slurm_logs/blip2-%j-out.txt
#SBATCH --error=./slurm_logs/blip2-%j-err.txt
#SBATCH --mem=64gb
#SBATCH --account=pasteur
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=pasteur
#SBATCH --time=192:00:00
#SBATCH --mail-user=jeffheo@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

eval "$(conda shell.bash hook)"
conda activate viberec

python blip2.py 