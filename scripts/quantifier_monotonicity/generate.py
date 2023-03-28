import os
from datasets import Dataset
from argparse import ArgumentParser
from multiprocessing import cpu_count
from ccg_parse import generate_samples

import json

def has_answer_entities(record):
    return record["answer_entities_count"] > 0

def expand_pair(batch):
    premises = []
    hypotheses = []
    labels = []

    for index, answer in enumerate(batch["answer"]):
        correct_answer = answer.replace("@placeholder", batch["correct_entity"][index])        
        results = generate_samples(correct_answer)

        hypotheses.extend([ result["hypothesis"] for result in results ])
        labels.extend([ str(result["label"]) for result in results ])
        premises.extend([ batch["reduced_summary"][index] ] * len(results))

    return {
        "premise": premises,
        "hypothesis": hypotheses,
        "label": labels
    }

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Augment data"
    )
    parser.add_argument('-i', '--input_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-s', '--start', required=True)
    parser.add_argument('-e', '--end', required=True)
    args = parser.parse_args()

    start_index = int(args.start)
    end_index = int(args.end)
    input_path = args.input_path
    if not os.path.exists(input_path):
        raise IOError("Dataset path does not exist")
    output_path = args.output_path
    if not os.path.exists(output_path):
        raise IOError("Output path does not exist")

    ds = Dataset.load_from_disk(dataset_path=input_path)
    augmented = ds.select(range(start_index, end_index+1)) \
        .filter(has_answer_entities, num_proc=cpu_count()) \
        .map(expand_pair, num_proc=cpu_count(), batched=True, remove_columns=ds.column_names)

    augmented.save_to_disk(dataset_path=f"{output_path}/chunk_{start_index}_{end_index}")