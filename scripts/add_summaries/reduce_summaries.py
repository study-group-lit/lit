import os
from datasets import Dataset, concatenate_datasets
from multiprocessing import cpu_count


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
    with_summaries = Dataset.from_dict(dict())
    for dir in os.walk("/workspace/students/lit/datasets/cnn_dataset_summaries"):
        chunk_path = dir[0]
        if chunk_path == "/workspace/students/lit/datasets/cnn_dataset_summaries":
            continue
        new_ds = Dataset.load_from_disk(chunk_path)
        with_summaries = concatenate_datasets([with_summaries, new_ds])

    with_reduced_summaries = with_summaries.map(reduce_summary, num_proc=cpu_count())
    with_reduced_summaries.save_to_disk(dataset_path="/workspace/students/lit/datasets/cnn_dataset_reduced_summaries")
