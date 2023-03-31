import os
import multiprocessing
from datasets import load_dataset
from transformers import pipeline

seeds = ["42", "69", "1337"]
cpu_count = multiprocessing.cpu_count()

model_path = "/mnt/semproj/sem_proj22/proj_05/data/models"
if not os.path.exists(model_path):
    raise IOError("Model path does not exist")
dataset_path = "/mnt/semproj/sem_proj22/proj_05/data/datasets"
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

def add_prediction_columns(records, model, seed):
    predictions = model([{"text": p, "text_pair": h} for p, h in zip(records["premise"], records["hypothesis"])])
    return { f"prediction_{seed}": [ label_mapping[p["label"]] for p in predictions ], f"score_{seed}": [ p["score"] for p in predictions ] }

for seed in seeds:
    print(f"Adding predictions of model {seed}...")
    model = models[seed]
    mnli["train"] = mnli["train"].map(add_prediction_columns, fn_kwargs={ "model": model, "seed": seed }, batched=True, batch_size=32)
mnli.save_to_disk(f"{dataset_path}/mnli_with_predictions")
