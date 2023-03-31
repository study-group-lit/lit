from transformers import pipeline
from evaluate import EvaluationSuite

model = pipeline("text-classification", model="/workspace/students/lit/models/roberta-base-finetuned-mnli/")

suite = EvaluationSuite.load("../../evaluation_suites/default.py")
print(suite.run(model))