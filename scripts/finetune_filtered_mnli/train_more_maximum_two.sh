#!/bin/bash
#SBATCH --job-name=finetune_more_filtered_maximum_two_mnli
#SBATCH --output=train_more_maximum_two.txt
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
python3 train_more.py -p /workspace/students/lit/models/roberta-base-finetuned-mnli-maximum_two/ -m /workspace/students/lit/models/roberta-base-finetuned-mnli-maximum_two_6/ -d /workspace/students/lit/datasets/mnli_maximum_two