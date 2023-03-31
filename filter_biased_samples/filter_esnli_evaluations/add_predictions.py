import os
import multiprocessing
from datasets import load_from_disk
import datasets
from transformers import pipeline

# Use this if the script crashes due to permissions problems with the cache
# datasets.disable_caching()

cpu_count = multiprocessing.cpu_count()

model_path = "/workspace/students/lit/models"
if not os.path.exists(model_path):
    raise IOError("Model path does not exist")
dataset_path = "/workspace/students/lit/datasets"
if not os.path.exists(dataset_path):
    raise IOError("Dataset path does not exist")

model = pipeline("text-classification", model=f"{model_path}/roberta-base-finetuned-mnli/")

# Currently we only have an evaluations and phenomena dataset for the finetuned mnli model for seed 42
esnli = load_from_disk("/workspace/students/lit/datasets/esnli_evaluations_42_phenomena")

label_mapping = {
    "ENTAILMENT": 0,
    "NEUTRAL": 1,
    "CONTRADICTION": 2
}

def add_prediction_columns(records, model):
    predictions = model([{"text": p, "text_pair": h} for p, h in zip(records["premise"], records["hypothesis"])])
    return { f"prediction": [ label_mapping[p["label"]] for p in predictions ], f"score": [ p["score"] for p in predictions ] }

print(f"Adding predictions...")
esnli = esnli.map(add_prediction_columns, fn_kwargs={ "model": model }, batched=True, batch_size=32)
esnli.save_to_disk(f"{dataset_path}/esnli_phenomena_with_predictions")
