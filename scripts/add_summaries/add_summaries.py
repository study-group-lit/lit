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

def add_summary_column(records, model: BartForConditionalGeneration, tokenizer: AutoTokenizer):
    texts = records["text"]
    inputs = tokenizer(texts, max_length=1024, truncation=True, padding=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=100)
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return { "summary": summaries }

if __name__ == "__main__":
    time.sleep(3600)

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

    bart_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")

    with_summaries = datasetdict_preprocessed.map(add_summary_column, fn_kwargs={"model": bart_model, "tokenizer": bart_tokenizer}, batched=True, batch_size=32)
    with_summaries.save_to_disk(dataset_dict_path=f"{dataset_path}_with_summaries")