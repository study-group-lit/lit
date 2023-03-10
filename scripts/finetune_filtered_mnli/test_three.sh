#!/bin/bash
#SBATCH --job-name=test_filtered_three_mnli
#SBATCH --output=test_three.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ~/lit/venv/bin/activate
python3 test.py -m /workspace/students/lit/models/roberta-base-finetuned-mnli-three/