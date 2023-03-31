#!/bin/bash
#SBATCH --job-name=test_maximum_one_3
#SBATCH --output=test_maximum_one_3.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 test.py -m /mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli-maximum_one_3/