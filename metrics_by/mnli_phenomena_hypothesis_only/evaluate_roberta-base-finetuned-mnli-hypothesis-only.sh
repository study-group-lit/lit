#!/bin/bash
#SBATCH --job-name=evaluate_roberta-base-finetuned-mnli-hypothesis-only
#SBATCH --output=evaluate_roberta-base-finetuned-mnli-hypothesis-only.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


#PATH=/opt/slurm/bin:$PATH


#source ../../venv/bin/activate
python evaluate_model.py -m ../../../lit-data/models/roberta-base-finetuned-mnli-hypothesis-only/42
