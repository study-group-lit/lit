from transformers import pipeline
from evaluate import EvaluationSuite

model = pipeline("text-classification", model="/workspace/students/lit/models/roberta-base-finetuned-recast/")

suite = EvaluationSuite.load("../../datasets/evaluation_suite.py")
print(suite.run(model))