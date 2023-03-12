#!/bin/bash
#SBATCH --job-name=finetune_filtered_three_mnli
#SBATCH --output=train_three.txt
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
python3 train.py -m /workspace/students/lit/models/roberta-base-finetuned-mnli-three/ -d /workspace/students/lit/datasets/mnli_three