import os
from datasets import DatasetDict
from argparse import ArgumentParser
from multiprocessing import cpu_count

all_quantifiers = {
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
}

def contains_quantifier_in(record, fields, all_quantifiers):
    for field in fields:
        for quantifier in all_quantifiers:
            if quantifier in record[field]:
                return True
    return False

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

    filtered = datasetdict.filter(contains_quantifier_in, fn_kwargs={"fields": ["answer"], "all_quantifiers": all_quantifiers}, num_proc=cpu_count())
    print(f"Size {filtered.num_rows}")
    filtered.save_to_disk(dataset_dict_path=f"{dataset_path}_filtered")