#!/bin/bash
#SBATCH --job-name=finetune_maximum_one_3
#SBATCH --output=finetune_maximum_one_3.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 train.py -m /mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli-maximum_one_3/ -d /mnt/semproj/sem_proj22/proj_05/data/datasets/mnli_maximum_one