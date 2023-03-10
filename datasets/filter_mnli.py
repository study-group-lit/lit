import os
import multiprocessing
from datasets import load_dataset, DatasetDict
from transformers import pipeline

seeds = ["42", "69", "1337"]
cpu_count = multiprocessing.cpu_count()

model_path = "/workspace/students/lit/models"
if not os.path.exists(model_path):
    raise IOError("Model path does not exist")
dataset_path = "/workspace/students/lit/datasets"
if not os.path.exists(dataset_path):
    raise IOError("Dataset path does not exist")

models = dict()
for seed in seeds:
    models[seed] = pipeline("text-classification", model=f"{model_path}/roberta-base-finetuned-mnli-hypothesis-only/{seed}")

mnli = load_dataset("multi_nli")

label_mapping = {
    "ENTAILMENT": 0,
    "NEUTRAL": 1,
    "CONTRADICTION": 2
}

def add_prediction_columns(record, model, seed):
    prediction = model({"text":record['premise'], "text_pair":record['hypothesis']})
    label = prediction["label"]
    score = prediction["score"]
    return { f"prediction_{seed}": label_mapping[label], f"score_{seed}": score }

for seed in seeds:
    print(f"Adding predictions of model {seed}...")
    model = models[seed]
    mnli["train"] = mnli["train"].map(add_prediction_columns, fn_kwargs={ "model": model, "seed": seed })
mnli.save_to_disk(f"{dataset_path}/mnli_with_predictions")

def correct_by_at_least(record, at_least):
    predictions = []
    for seed in seeds:
        predictions.append(record[f"prediction_{seed}"])
    return len(list(filter(lambda pred: pred == record["label"], predictions))) >= at_least

mnli_at_least_two = DatasetDict()
print(f"Filtering split by at least two...")
mnli_at_least_two["train"] = mnli["train"].filter(correct_by_at_least, fn_kwargs={ "at_least": 2 }, num_proc=cpu_count)
print(f"Saving filtered datset...")
mnli_at_least_two.save_to_disk(f"{dataset_path}/mnli_at_least_two")

mnli_three = DatasetDict()
print(f"Filtering split by three...")
mnli_three["train"] = mnli["train"].filter(correct_by_at_least, fn_kwargs={ "at_least": 3 }, num_proc=cpu_count)
print(f"Saving filtered datset...")
mnli_three.save_to_disk(f"{dataset_path}/mnli_three")
