import os
from datasets import load_from_disk, concatenate_datasets, ClassLabel, Value
from evaluate import EvaluationSuite
from argparse import ArgumentParser
import json
from multiprocessing import cpu_count
import datasets

# Use this if the script crashes due to permissions problems with the cache
datasets.disable_caching()

cnn_path = "/workspace/students/lit/datasets/cnn_dataset_augmented"
dailymail_path = "/workspace/students/lit/datasets/dailymail_dataset_augmented"

LABELS = ["entailment", "neutral", "contradiction"]


def count_label_occurence(dataset, label):
    return dataset.filter(lambda record: record["label"] == label, num_proc=cpu_count()).num_rows

if not os.path.exists(cnn_path) or not os.path.exists(dailymail_path):
    raise IOError("Dataset path does not exist")

cnn = load_from_disk(cnn_path)
dailymail = load_from_disk(dailymail_path)
recast = concatenate_datasets([cnn, dailymail])
dataset = recast.cast_column("label", Value("int64")).cast_column("label", ClassLabel(names= [
    "entailment",
    "neutral",
    "contradiction"
]))

splits = {"validation": dataset.num_rows}
classes = {
    name: count_label_occurence(dataset, index) for index, name in enumerate(LABELS)
}
print("Statistics for recast dataset")
print(json.dumps(splits))
print(json.dumps(classes))
print("")
with open("./results/recast.json", "w") as f:
    f.write(json.dumps(
        {
            "splits": splits,
            "classes": classes
        }
    ))
