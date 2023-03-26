import os
from datasets import load_from_disk
from evaluate import EvaluationSuite
from argparse import ArgumentParser
import json
from multiprocessing import cpu_count
import datasets

# Use this if the script crashes due to permissions problems with the cache
datasets.disable_caching()

DATASET_PREFIX = "/workspace/students/lit/datasets/"
DATASETS = [
    DATASET_PREFIX + name for name in [
        "esnli_phenomena_with_predictions_filtered" # ESNLI samples, which were predicted by the model
    ]
]

LABELS = ["entailment", "neutral", "contradiction"]


def count_label_occurence(dataset, label):
    return len(dataset.filter(lambda record: record["label"] == label, num_proc=cpu_count()))

def calculate_dataset_counts(dataset_path):
    if not os.path.exists(dataset_path):
        raise IOError("Dataset path does not exist")

    dataset = load_from_disk(dataset_path)
    splits = {"validation": dataset.num_rows}
    classes = {
        name: count_label_occurence(dataset, index) for index, name in enumerate(LABELS)
    }
    print("Statistics for dataset at " + dataset_path)
    print(json.dumps(splits))
    print(json.dumps(classes))
    print("")
    with open("./results/" + dataset_path.split("/")[-1] + ".json", "w") as f:
        f.write(json.dumps(
            {
                "splits": splits,
                "classes": classes
            }
        ))

for dataset_path in DATASETS:
    calculate_dataset_counts(dataset_path)
