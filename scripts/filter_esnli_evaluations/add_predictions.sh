#!/bin/bash
#SBATCH --job-name=add_predictions
#SBATCH --output=add_predictions.txt
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
python3 add_predictions.py