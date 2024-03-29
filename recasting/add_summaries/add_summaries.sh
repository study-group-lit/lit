#!/bin/bash
#SBATCH --job-name=add_summaries_dailymail_37001_39000
#SBATCH --output=add_summaries_dailymail_37001_39000.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 add_summaries.py -d /mnt/semproj/sem_proj22/proj_05/data/datasets/dailymail_dataset_filtered_filtered_pos -o /mnt/semproj/sem_proj22/proj_05/data/datasets/dailymail_dataset_filtered_pos_summaries -p training -s 37001 -e 39000