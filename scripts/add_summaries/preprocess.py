import os
import re
from datasets import DatasetDict
from multiprocessing import cpu_count
from transformers import AutoTokenizer, BartForConditionalGeneration
from argparse import ArgumentParser
import time

def parse_entity_mapping(mapping_str):
    mappings_list = mapping_str.split("\n")
    mappings = dict()
    for mapping in mappings_list:
        mapping_splits = mapping.split(":")
        mappings[mapping_splits[0]] = mapping_splits[1]
    return mappings

def preprocess(record):
    entity_mapping = parse_entity_mapping(record["entity_mapping"])

    # unmask text
    text = record["text"]
    for placeholder, entity in entity_mapping.items():
        text = re.sub(placeholder, entity, text)
    split_text = text.split(" ")
    source = split_text[1]
    text = " ".join(split_text[3:])

    # unmask correct entity
    correct_entity = entity_mapping[record["correct_entity"]]

    # get entites in answer
    answer_entites_list = []
    answer = record["answer_masked"]
    split_answer = answer.split(" ")
    for token in split_answer:
        if token.startswith("@"):
            try:
                answer_entites_list.append(entity_mapping[token])
            except KeyError:
                pass
    answer_entities = " ".join(answer_entites_list)

    # unmask answer
    answer = record["answer_masked"]
    for placeholder, entity in entity_mapping.items():
        answer = re.sub(placeholder, entity, answer)

    return {"text": text, "correct_entity": correct_entity, "source": source, "answer_entities": answer_entities, "answer": answer }

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

    datasetdict_preprocessed = datasetdict.map(preprocess, num_proc=cpu_count())
    datasetdict_preprocessed.save_to_disk(dataset_dict_path=f"{dataset_path}_preprocessed")
