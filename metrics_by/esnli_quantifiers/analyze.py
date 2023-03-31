# README
# This file contains a script to test the performance of a model using the EvaluationSuites defined in 
# 1. ../../evaluation_suites/esnli_quantifiers_hypothesis_only.py for hypothesis-only models
# 2. ../../evaluation_suites/esnli_quantifiers.py for regular models.
# Tests will run on samples form e-SNLI that contain quantifiers and all metrics will be calculated 
# separated by concrete quantifiers occuring in the samples.

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

evaluation_suite_path = "../../evaluation_suites/esnli_quantifiers_hypothesis_only.py" if hypothesis_only else "../../evaluation_suites/esnli_quantifiers.py"
suite = EvaluationSuite.load(evaluation_suite_path)
results = suite.run(model)

for result in results:
    result["task_name"] = result["task_name"].info.description
print(results)