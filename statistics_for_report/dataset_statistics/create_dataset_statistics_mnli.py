# README
# Calculates the following two statistics over the Filtered 2/2 and Filtered 3/2 datasets:
# 1. Sample count for each of the labels
# 2. Sample count per split
# The values are printed as JSON and also stored in the ./results folder

import os
from datasets import load_from_disk
from evaluate import EvaluationSuite
from argparse import ArgumentParser
import json
from multiprocessing import cpu_count
import datasets

# Use this if the script crashes due to permissions problems with the cache
datasets.disable_caching()

DATASET_PREFIX = "/mnt/semproj/sem_proj22/proj_05/data/datasets/"
DATASETS = [
    DATASET_PREFIX + name
    for name in [
        "mnli_maximum_one",  # MultiNLI samples, which were predicted by a maximum of 1 / 2 out of the three models
        "mnli_maximum_two",
    ]
]

LABELS = ["entailment", "neutral", "contradiction"]


def count_label_occurence(dataset, label, splits):
    return sum(
        [
            len(
                dataset[split].filter(
                    lambda record: record["label"] == label, num_proc=cpu_count()
                )
            )
            for split, _ in splits.items()
        ]
    )


def calculate_dataset_counts(dataset_path):
    if not os.path.exists(dataset_path):
        raise IOError("Dataset path does not exist")

    dataset = load_from_disk(dataset_path)
    splits = dataset.num_rows
    classes = {
        name: count_label_occurence(dataset, index, splits)
        for index, name in enumerate(LABELS)
    }
    print("Statistics for dataset at " + dataset_path)
    print(json.dumps(splits))
    print(json.dumps(classes))
    print("")
    with open("./results/" + dataset_path.split("/")[-1] + ".json", "w") as f:
        f.write(json.dumps({"splits": splits, "classes": classes}))


for dataset_path in DATASETS:
    calculate_dataset_counts(dataset_path)
