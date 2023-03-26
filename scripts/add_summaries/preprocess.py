import os
from datasets import DatasetDict
from multiprocessing import cpu_count
from argparse import ArgumentParser

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
    split_text = text.split(" ")

    source = split_text[1]

    split_text_unmasked = []
    for token in split_text:
        if token.startswith("@"):
            try:
                split_text_unmasked.append(entity_mapping[token])
            except KeyError:
                split_text_unmasked.append(token)
        else:
            split_text_unmasked.append(token)
    text = " ".join(split_text_unmasked)

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
    split_answer = answer.split(" ")
    split_answer_unmasked = []
    for token in split_answer:
        if token.startswith("@"):
            try:
                split_answer_unmasked.append(entity_mapping[token])
            except KeyError:
                split_answer_unmasked.append(token)
        else:
            split_answer_unmasked.append(token)
    answer = " ".join(split_answer_unmasked)

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

    datasetdict_preprocessed = datasetdict["training"].select(range(0, 10)).map(preprocess, num_proc=cpu_count())
    # datasetdict_preprocessed.save_to_disk(dataset_dict_path=f"{dataset_path}_preprocessed")
    datasetdict_preprocessed.to_csv("/home/students/trump/lit/test.csv")
