#!/bin/bash
#SBATCH --job-name=add_summaries_cnn_39001_42000
#SBATCH --output=add_summaries_cnn_39001_42000.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu


PATH=/opt/slurm/bin:$PATH


source ~/lit/venv/bin/activate
python3 add_summaries.py -d /workspace/students/lit/datasets/cnn_dataset_preprocessed_filtered -p training -s 39001 -e 42000