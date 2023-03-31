# README
# Calculate metrics on the following evaluation suites found in ../../evaluation_suites:
# 1. Metrics on SICK test
# 2. Metrics on e-SNLI test for each phenomena
# for each of the following models:
# 1. Base MultiNLI
# 2. Recast
# 3. All filtered models
# 4. Ensemble
# 5. All three hypothesis only models
# The hypothesis only models require a slightly different evaluation suite, as the models only receive the hypothesis

from transformers import pipeline
from evaluate import EvaluationSuite
import datasets
datasets.disable_caching()

models = [
    "roberta-base-finetuned-mnli",
    "roberta-base-finetuned-recast",
    "roberta-base-finetuned-mnli-maximum_one_3",
    "roberta-base-finetuned-mnli-maximum_one_6",
    "roberta-base-finetuned-mnli-maximum_two_3",
    "roberta-base-finetuned-mnli-maximum_two_6",
    "roberta-ensemble-finetuned-mnli_0_03"
]

models = ["/mnt/semproj/sem_proj22/proj_05/data/models/" + model for model in models]

for model in models:
    print(model)
    model = pipeline("text-classification", model=model)
    print("sick test")
    suite = EvaluationSuite.load("../../evaluation_suites/sick.py")
    print(suite.run(model))
    print("esnli test")
    suite = EvaluationSuite.load("../../evaluation_suites/esnli_test_phenomena.py")
    results = suite.run(model)
    for result in results:
        result["task_name"] = result["task_name"].info.description
    print(results)
    print("")


model = "/mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli-hypothesis-only/"
seeds = [
    "42", "69", "1337"
]
for seed in seeds:
    print(model+seed)
    model = pipeline("text-classification", model=model+seed)
    print("sick test")
    suite = EvaluationSuite.load("../../evaluation_suites/sick_hypothesis_only.py")
    print(suite.run(model))
    print("esnli test")
    suite = EvaluationSuite.load("../../evaluation_suites/esnli_test_phenomena_hypothesis_only.py")
    results = suite.run(model)
    for result in results:
        result["task_name"] = result["task_name"].info.description
    print(results)
    print("")
