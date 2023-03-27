import os
from datasets import DatasetDict
from transformers import pipeline
from argparse import ArgumentParser

def add_summary_column(records, model):
    texts = records["text"]
    # 1024 was taken from https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration.forward.example
    summaries = model([text[:1024] for text in texts])
    summaries = list(map(lambda summary_dict: summary_dict["summary_text"], summaries))
    return { "summary": summaries }

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Add summaries"
    )
    parser.add_argument('-d', '--dataset_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-p', '--split', required=True)
    parser.add_argument('-s', '--start', required=True)
    parser.add_argument('-e', '--end', required=True)
    args = parser.parse_args()

    start_index = int(args.start)
    end_index = int(args.end)
    split = args.split
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        raise IOError("Dataset path does not exist")
    output_path = args.output_path
    if not os.path.exists(output_path):
        raise IOError("Output path does not exist")

    datasetdict = DatasetDict.load_from_disk(dataset_path)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

    with_summaries = datasetdict[split].select(range(start_index, end_index+1)).map(add_summary_column, fn_kwargs={"model": summarizer}, batched=True, batch_size=32)
    with_summaries.save_to_disk(dataset_path=f"{output_path}/{split}_chunk_{start_index}_{end_index}")