#!/bin/bash
#SBATCH --job-name=finetune_recast
#SBATCH --output=train.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu
#SBATCH --nodelist=gpu08


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 train.py