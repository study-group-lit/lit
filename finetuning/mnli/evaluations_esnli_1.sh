#!/bin/bash
#SBATCH --job-name=calculate_evaluations_on_esnli_1
#SBATCH --output=calculate_evaluations_on_esnli_1.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=32000
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ../../venv/bin/activate
python3 calculate_evaluations_on_esnli.py -c 1