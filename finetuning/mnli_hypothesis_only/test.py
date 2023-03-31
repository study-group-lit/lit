# README
# This file contains a script to test the performance a model trained only 
# on the hypotheses of MultiNLI using the  EvaluationSuite defined 
# in ../../evaluation_suites/default.py.

from transformers import pipeline
from evaluate import EvaluationSuite
import argparse

parser = argparse.ArgumentParser(
                    prog = 'Hypothesis only MNLI tests',
                    description = 'Tests RoBERTa using only MNLI hypothesis')
parser.add_argument("seed", default=42, type=int, nargs='?')
args = parser.parse_args()

seed = args.seed
model = pipeline("text-classification", model=f"/mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli-hypothesis-only/{seed}/")

suite = EvaluationSuite.load("../../evaluation_suites/mnli_hypothesis_only.py")
print(suite.run(model))