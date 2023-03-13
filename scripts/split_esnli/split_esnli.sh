#!/bin/bash
#SBATCH --job-name=split_esnli
#SBATCH --output=split_esnli.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch


PATH=/opt/slurm/bin:$PATH


source ~/lit/venv/bin/activate
python3 split_esnli.py