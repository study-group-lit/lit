from transformers import pipeline
from evaluate import EvaluationSuite

model = pipeline("text-classification", model="/workspace/students/lit/models/roberta-ensemble-finetuned-mnli/")

suite = EvaluationSuite.load("../../datasets/evaluation_suite.py")
print(suite.run(model))