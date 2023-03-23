#!/bin/bash
#SBATCH --job-name=evaluations_filtered_esnli
#SBATCH --output=evaluations_filtered_esnli.txt
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
python3 evaluations_filtered_esnli.py
