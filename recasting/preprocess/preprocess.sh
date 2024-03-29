#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch

PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 preprocess.py -d /mnt/semproj/sem_proj22/proj_05/data/datasets/cnn_dataset