from datasets import load_dataset
from transformers import pipeline
from evaluate import EvaluationSuite

model = pipeline("text-classification", model="../../models/roberta-base-mnli/")

suite = EvaluationSuite.load("../../datasets/test_suite.py")
print(suite.run(model))