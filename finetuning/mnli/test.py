# README
# This file contains a script to test the performance of a model using the 
# EvaluationSuite defined in ../../evaluation_suites/default.py.

from transformers import pipeline
from evaluate import EvaluationSuite

model = pipeline("text-classification", model="/mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli/")

suite = EvaluationSuite.load("../../evaluation_suites/default.py")
print(suite.run(model))