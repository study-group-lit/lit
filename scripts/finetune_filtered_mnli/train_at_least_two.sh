#!/bin/bash
#SBATCH --job-name=finetune_filtered_at_least_two_mnli
#SBATCH --output=train_at_least_two.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ~/lit/venv/bin/activate
python3 train.py -m /workspace/students/lit/models/roberta-base-finetuned-mnli-at-least-two/ -d /workspace/students/lit/datasets/mnli_at_least_two