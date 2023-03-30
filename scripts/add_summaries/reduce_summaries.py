import os
from datasets import Dataset, concatenate_datasets
from multiprocessing import cpu_count
from argparse import ArgumentParser


def reduce_summary(record):
    sentences = record["summary"].split(" . ")
    answer_entites = record["answer_entities"].split(",")
    sentences_count = dict()
    for sentence in sentences:
        tokens = sentence.split(" ")
        count_answer_entites = len(list(filter(lambda token: token in answer_entites, tokens)))
        sentences_count[sentence] = count_answer_entites
    sentence_maximum_count = max(sentences_count, key=sentences_count.get)
    return { "reduced_summary": sentence_maximum_count, "answer_entities_count": sentences_count[sentence_maximum_count] }

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Add summaries"
    )
    parser.add_argument('-i', '--input_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        raise IOError("Input path does not exist")
    output_path = args.output_path

    with_summaries = Dataset.from_dict(dict())
    for dir in os.walk(input_path):
        chunk_path = dir[0]
        if chunk_path == input_path:
            continue
        new_ds = Dataset.load_from_disk(chunk_path)
        with_summaries = concatenate_datasets([with_summaries, new_ds])

    with_reduced_summaries = with_summaries.map(reduce_summary, num_proc=cpu_count())
    with_reduced_summaries.save_to_disk(dataset_path=output_path)
