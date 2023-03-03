#!/bin/bash
#SBATCH --job-name=finetune_mnli_hypothesis_only
#SBATCH --output=train_42.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=32000
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ~/lit/venv/bin/activate
python3 train.py 42