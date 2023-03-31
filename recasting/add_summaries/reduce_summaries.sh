#!/bin/bash
#SBATCH --job-name=reduce_cnn_summaries
#SBATCH --output=reduce_cnn_summaries.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch

PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 reduce_summaries.py -i /mnt/semproj/sem_proj22/proj_05/data/datasets/cnn_dataset_fixed_summaries -o /mnt/semproj/sem_proj22/proj_05/data/datasets/cnn_dataset_fixed_summaries_reduced