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

models = ["/workspace/students/lit/models/" + model for model in models]

for model in models:
    print(model)
    model = pipeline("text-classification", model=model)
    print("sick test")
    suite = EvaluationSuite.load("../../datasets/evaluation_suite_sick_test.py")
    print(suite.run(model))
    print("esnli test")
    suite = EvaluationSuite.load("../../datasets/evaluation_suite_phenomena_test.py")
    results = suite.run(model)
    for result in results:
        result["task_name"] = result["task_name"].info.description
    print(results)
    print("")


model_path = "/workspace/students/lit/models/roberta-base-finetuned-mnli-hypothesis-only/"

seeds = [
    "42", "69", "1337"
]
for seed in seeds:
    print(model_path+seed)
    model = pipeline("text-classification", model=model_path+seed)
    print("sick test")
    suite = EvaluationSuite.load("../../datasets/evaluation_suite_sick_test_hypothesis_only.py")
    print(suite.run(model))
    print("esnli test")
    suite = EvaluationSuite.load("../../datasets/evaluation_suite_phenomena_hypothesis_only_test.py")
    results = suite.run(model)
    for result in results:
        result["task_name"] = result["task_name"].info.description
    print(results)
    print("")