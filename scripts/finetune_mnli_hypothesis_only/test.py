from transformers import pipeline
from evaluate import EvaluationSuite
import argparse

parser = argparse.ArgumentParser(
                    prog = 'Hypothesis only MNLI tests',
                    description = 'Tests RoBERTa using only MNLI hypothesis')
parser.add_argument("seed", default=42, type=int, nargs='?')
args = parser.parse_args()

seed = args.seed
model = pipeline("text-classification", model=f"/workspace/students/lit/models/roberta-base-finetuned-mnli-hypothesis-only/{seed}/")

suite = EvaluationSuite.load("../../datasets/evaluation_suite.py")
print(suite.run(model))