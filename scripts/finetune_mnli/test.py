from datasets import load_dataset
from transformers import pipeline
from evaluate import evaluator
import evaluate

sick = load_dataset("sick")
mnli = load_dataset("multi_nli")

model = pipeline("text-classification", model="../../models/roberta-base-mnli/")

metrics = evaluate.combine(["accuracy", "matthews_correlation"])

sick_validation =  sick["validation"]
mnli_m = mnli["validation_matched"]
mnli_mm = mnli["validation_mismatched"]

def evaluate(data, premise="premise", hypothesis="hypothesis"):
    return evaluator("text-classification") \
        .compute(
            model,
            data,
            metric=metrics,
            label_mapping={"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2},
            input_column=premise,
            second_input_column=hypothesis
        )

print(evaluate(sick_validation, "sentence_A", "sentence_B"))
print(evaluate(mnli_m))
print(evaluate(mnli_mm))
