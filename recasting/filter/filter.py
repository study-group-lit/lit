# README
# This file contains a script to filter samples form the CNN and Dailymail dataset 
# that contain quantifiers in the answer based on pure string matching. 

import os
from datasets import DatasetDict
from argparse import ArgumentParser
from multiprocessing import cpu_count
from nltk import word_tokenize, ngrams
import nltk

nltk.download('punkt')

all_quantifiers = [
    "a few",
    "a large number of",
    "a little",
    "a number of",
    "a small number of",
    "all",
    "any",
    "enough", 
    "each",
    "every",
    "few",
    "fewer",
    "less",
    "lots of",
    "many",
    "most",
    "much",
    "no",
    "none of",
    "not many",
    "not much",
    "numerous",
    "plenty of",
    "several",
    "some",
    "whole",
    "many of"
]

def contains_quantifier_in_answer(record):
    shingles = []
    tokens = word_tokenize(record["answer"])
    for n in range(1, 5):
        shingles.extend(ngrams(tokens, n))
    shingles = map(lambda shingle: " ".join(shingle), shingles)
    quantifiers = list(filter(lambda shingle: shingle in all_quantifiers, shingles))
    return len(quantifiers) > 0

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Add summaries"
    )
    parser.add_argument('-d', '--dataset_path', required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        raise IOError("Dataset path does not exist")

    datasetdict = DatasetDict.load_from_disk(dataset_path)

    filtered = datasetdict.filter(contains_quantifier_in_answer, num_proc=cpu_count())
    print(f"Size {filtered.num_rows}")
    filtered.save_to_disk(dataset_dict_path=f"{dataset_path}_filtered")