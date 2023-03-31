#!/bin/bash
#SBATCH --job-name=filter_pos_dailymail
#SBATCH --output=filter_pos_dailymail.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch

PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 filter_pos.py -d /mnt/semproj/sem_proj22/proj_05/data/datasets/dailymail_dataset_filtered