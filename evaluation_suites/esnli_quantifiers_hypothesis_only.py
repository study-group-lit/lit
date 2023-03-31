# README
# This file contains an EvaluationSuite to evaluate the performance of a hypothesis-only 
# model on the part of the validation split of e-SNLI that contain quantifiers in the words 
# of premise and hypothesis that have been marked as important by human annotators. The results
# are split by quantifiers.

import evaluate
from evaluate.evaluation_suite import SubTask
from multiprocessing import cpu_count
from datasets import DatasetDict
import re
import datasets
datasets.disable_caching()


def filter_quantifiers(dataset):
    dataset["validation"] = dataset["validation"].filter(
        lambda record: record["quantifiers"] > 0)
    return dataset


def split_consecutive_sublists(indices):
    if len(indices) == 0:
        return []
    indices.sort()
    sublists = []
    current_sublist = [indices[0]]
    for item in indices[1:]:
        if current_sublist[-1] == item-1:
            current_sublist.append(item)
        else:
            sublists.append(current_sublist)
            current_sublist = [item]
    return sublists


def parse_indices(indices: str):
    if indices in ["{}", ""]:
        return []
    return [int(i) for i in indices.split(",")]


def get_words_at_indices(string: str, indices):
    split_string = string.split(" ")
    # Remove punctuation marks and empty strings
    return list(filter(lambda word: word != "", map(lambda i: re.sub(r"[.,!?]", "", split_string[i]), indices)))


def get_important_words_grouped(record: dict, hypothesis_only: bool):
    words_and_groups = set()
    parts = ["hypothesis"] if hypothesis_only else ["premise", "hypothesis"]
    for sentence in parts:
        for i in range(1, 4):
            word_indices_premise = parse_indices(
                record[f"{sentence}_highlighted_{i}"])
            grouped_word_indices_premise = split_consecutive_sublists(
                word_indices_premise)
            for word_group_indices in grouped_word_indices_premise:
                word_group = get_words_at_indices(
                    record[sentence], word_group_indices)
                words_and_groups.update(word_group)
                words_and_groups.update(" ".join(word_group))
    return words_and_groups


def has_quantifier(record: str, quantifier: str, hypothesis_only: bool) -> bool:
    important_words_grouped = get_important_words_grouped(
        record, hypothesis_only)
    return any([phrase == quantifier for phrase in important_words_grouped])

# code from https://discuss.huggingface.co/t/combining-metrics-for-multiclass-predictions-evaluations/21792/11


class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs

    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)

    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()


class Suite(evaluate.EvaluationSuite):
    def __init__(self, name):
        super().__init__(name)
        esnli_phenomena = DatasetDict.load_from_disk("/mnt/semproj/sem_proj22/proj_05/data/datasets/esnli_phenomena")
        self.esnli_quantifiers = filter_quantifiers(esnli_phenomena)
        self.quantifiers = {
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
        self.suite = list(filter(None, [self.task_for(
            quantifier, True) for quantifier in self.quantifiers]))

    def task_for(self, quantifier: str, hypothesis_only: bool):
        metric = evaluate.combine([
            evaluate.load("accuracy"),
            evaluate.load("matthews_correlation"),
            ConfiguredMetric(evaluate.load("f1"), average="macro"),
            evaluate.load("BucketHeadP65/confusion_matrix")
        ])

        print(f"Filtering data by quantifier {quantifier}...")
        data = self.esnli_quantifiers["validation"].filter(
            lambda r: has_quantifier(r, quantifier, hypothesis_only), num_proc=cpu_count())
        data.info.description = quantifier
        print(f"Dataset for {quantifier} has size {data.num_rows}...")

        return None if data.num_rows == 0 else SubTask(
            task_type="text-classification",
            data=data,
            args_for_task={
                "metric": metric,
                "input_column": "hypothesis",
                "label_column": "label",
                "label_mapping": {
                    "ENTAILMENT": 0,
                    "NEUTRAL": 1,
                    "CONTRADICTION": 2
                }
            }
        )
