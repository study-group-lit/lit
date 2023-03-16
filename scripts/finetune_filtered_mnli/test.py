import os
from argparse import ArgumentParser
from transformers import pipeline
from evaluate import EvaluationSuite

parser = ArgumentParser(
    prog="Train"
)
parser.add_argument('-m', '--model_path', required=True)
args = parser.parse_args()

# check paths and create if needed
model_path = args.model_path
if not os.path.exists(model_path):
    raise IOError("Model path does not exist")

model = pipeline("text-classification", model=model_path)

suite = EvaluationSuite.load("../../datasets/evaluation_suite.py")
print(suite.run(model))