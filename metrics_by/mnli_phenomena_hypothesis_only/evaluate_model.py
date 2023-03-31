import os
from transformers import pipeline
from evaluate import EvaluationSuite
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Evaluate by phenomena"
)
parser.add_argument('-m', '--model_path', required=True)
args = parser.parse_args()

# check paths and create if needed
model_path = args.model_path
if not os.path.exists(model_path):
    raise IOError("Model path does not exist")

model = pipeline("text-classification", model=model_path)

suite = EvaluationSuite.load("../../datasets/evaluation_suites/mnli_phenomena_hypothesis_only.py")
results = suite.run(model)

for result in results:
    result["task_name"] = result["task_name"].info.description
print(results)