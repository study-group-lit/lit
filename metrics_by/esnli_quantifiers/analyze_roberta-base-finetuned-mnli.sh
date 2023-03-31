#!/bin/bash
#SBATCH --job-name=analyze_roberta-base-finetuned-mnli
#SBATCH --output=analyze_roberta-base-finetuned-mnli.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 analyze.py -m /mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli