import os
from transformers import pipeline
from evaluate import EvaluationSuite
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Evaluate by phenomena"
)
parser.add_argument('-m', '--model_path', required=True)
parser.add_argument("-o", "--hypothesis_only", action='store_true')
args = parser.parse_args()

# check paths and create if needed
model_path = args.model_path
hypothesis_only = args.hypothesis_only
if not os.path.exists(model_path):
    raise IOError("Model path does not exist")

model = pipeline("text-classification", model=model_path)

evaluation_suite_path = "../../evaluation_suites/esnli_phenomena_hypothesis_only.py" if hypothesis_only else "../../evaluation_suites/esnli_phenomena.py"
suite = EvaluationSuite.load(evaluation_suite_path)
results = suite.run(model)

for result in results:
    result["task_name"] = result["task_name"].info.description
print(results)