# README
# This file contains a script to filter samples form the CNN and Dailymail dataset 
# that contain quantifiers in the answer based on POS tagging with nltk. 

import os

from typing import List
from dataclasses import dataclass

from datasets import DatasetDict
from argparse import ArgumentParser
from multiprocessing import cpu_count
from nltk import word_tokenize, pos_tag
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

@dataclass
class Quantifier:
    name: str
    left_rising: bool
    right_rising: bool
    tags: List[List[str]]

all_quantifiers = [
    Quantifier("a few", True, True, [["DT", "JJ"]]),
    Quantifier("a large number of", True, True, [["DT", "JJ", "NN", "IN"]]),
    Quantifier("a little", True, True, [["DT", "JJ"]]),
    Quantifier("a number of", True, True, [["DT", "NN", "IN"]]),
    Quantifier("a small number of", True, True, [["DT", "JJ", "NN", "IN"]]),
    Quantifier("all", False, True, [["DT"]]),
    Quantifier("any", False, True, [["DT"]]),
    Quantifier("enough", True, True, [["DT"]]),
    Quantifier("each", False, True, [["DT"]]),
    Quantifier("every", True, True, [["DT"]]),
    Quantifier("few", False, False, [["DT"]]),
    Quantifier("fewer", False, False, [["DT"]]),
    Quantifier("less", False, False, [["DT"], ["RB"], ["IN"], ["JJR"]]), # Also adverb and preposition
    Quantifier("lots of", True, True, [["RB", "IN"], ["NNS", "IN"]]), # Idiom: adverb + preposition
    Quantifier("many", True, True, [["DT"], ["JJ"]]),
    Quantifier("most", False, True, [["DT"]]),
    Quantifier("most of", False, True, [["JJS", "IN"]]),
    Quantifier("much", True, True, [["DT"]]),
    Quantifier("much of", True, True, [["NN", "IN"]]),
    Quantifier("no", False, False, [["DT"]]),
    Quantifier("none of", False, False, [["NN", "IN"]]),
    Quantifier("not many", False, False, [["RB", "JJ"]]),
    Quantifier("not much", False, False, [["RB", "JJ"]]),
    Quantifier("numerous", True, True, [["JJ"]]), # Adjective
    Quantifier("plenty of", True, True, [["NN", "IN"]]), # Idiom: Pronoun + preposition
    Quantifier("several", True, True, [["DT"], ["JJ"]]), # Also pronoun
    Quantifier("some", True, True, [["DT"]]),
    Quantifier("whole", False, True, [["RB"], ["JJ"]]), # Adverb
    Quantifier("many of", True, True, [["NN", "IN"]]), # Noun + preposition
]


def contains_correct_quantifier_in_answer(record):
    tokens = pos_tag(word_tokenize(record["answer"]))
    for quantifier in all_quantifiers:
        s = " ".join(word for word, _ in tokens)
        if quantifier.name + " " not in s:
            continue
        match = s.find(quantifier.name + " ") # character index
        match = len(word_tokenize(s[:match])) # token index
        word_tags = [pos for _, pos in tokens][match:]
        for tags in quantifier.tags:
            if all(t1 == t2 for t1, t2 in zip(tags, word_tags)):
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

    filtered = datasetdict.filter(contains_correct_quantifier_in_answer, num_proc=cpu_count())
    print(f"Size {filtered.num_rows}")
    filtered.save_to_disk(dataset_dict_path=f"{dataset_path}_filtered_pos")