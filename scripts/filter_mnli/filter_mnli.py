import os
import multiprocessing
import copy
from datasets import load_from_disk

seeds = ["42", "69", "1337"]
cpu_count = multiprocessing.cpu_count()

dataset_path = "/workspace/students/lit/datasets"
if not os.path.exists(dataset_path):
    raise IOError("Dataset path does not exist")

mnli = load_from_disk(dataset_path)

def correct_by_at_least(record, at_least):
    predictions = []
    for seed in seeds:
        predictions.append(record[f"prediction_{seed}"])
    return len(list(filter(lambda pred: pred == record["label"], predictions))) >= at_least

def correct_by_maximum(record, maximum):
    predictions = []
    for seed in seeds:
        predictions.append(record[f"prediction_{seed}"])
    return len(list(filter(lambda pred: pred == record["label"], predictions))) <= maximum

mnli_maximum_one = copy.deepcopy(mnli)
# throw out 2 or 3 correct -> keep 0, 1 correct
print(f"Filtering split by maximum one...")
mnli_maximum_one["train"] = mnli_maximum_one["train"].filter(correct_by_maximum, fn_kwargs={ "maximum": 1 }, num_proc=cpu_count)
print(f"Saving filtered datset...")
mnli_maximum_one.save_to_disk(f"{dataset_path}/mnli_maximum_one")

mnli_maximum_two = copy.deepcopy(mnli)
# throw out 3 correct -> keep 0, 1 or 2 correct
print(f"Filtering split by maximum two...")
mnli_maximum_two["train"] = mnli_maximum_two["train"].filter(correct_by_maximum, fn_kwargs={ "maximum": 2 }, num_proc=cpu_count)
print(f"Saving filtered datset...")
mnli_maximum_two.save_to_disk(f"{dataset_path}/mnli_maximum_two")
