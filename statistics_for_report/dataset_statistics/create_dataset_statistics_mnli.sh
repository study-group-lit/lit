#!/bin/bash
#SBATCH --job-name=create_dataset_statistics_mnli
#SBATCH --output=create_dataset_statistics_mnli.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 create_dataset_statistics_mnli.py