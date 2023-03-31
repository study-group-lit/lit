# README
# This file contains a script to get the number of occurences of linguisitc phenomena in a dataset
# where the linguisitc phenomena have been detected using the script in ./split_dataset.py,
# ./split_dataset_hypothesis_only.py or ../esnli/split_esnli.py

from datasets import DatasetDict
from multiprocessing import cpu_count
import json
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Get occurences of phenomena"
)
parser.add_argument('-d', '--dataset_path', required=True)
args = parser.parse_args()

dataset = DatasetDict.load_from_disk(args.dataset_path)
phenomena = ["synonym", "antonym", "hypernym", "hyponym", "co_hyponym", "quantifiers", "numericals"]
splits = dataset.keys()
occurences = { split: [] for split in splits }

for split in splits:
    for phenomenon in phenomena:
        filtered = dataset[split].filter(lambda r: r[phenomenon] > 0, num_proc=cpu_count())
        occurences[split].append({
            "phenomenon": phenomenon,
            "count": filtered.num_rows
        })
print(json.dumps(occurences, indent=4))