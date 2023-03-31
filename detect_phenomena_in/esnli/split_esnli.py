# README
# This file contains a script to detect the liguistic phenomena of synonym, antonym,
# hypernym, hyponym, co_hyponym, quantifiers and numerals in in the words marked as 
# important by human annotators in the e-SNLI dataset. For each phenomenon a column 
# will be added to the dataset which contains a number. If the number is >0, the 
# respective phenomenon has been found in the sample.

from datasets import load_dataset
import re
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
import multiprocessing
import warnings
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

num_cpus = multiprocessing.cpu_count()
esnli = load_dataset("../../datasets/esnli.py")

def transform_highlighted(record):
    highlighted_premise_all = set()
    highlighted_hypothesis_all = set()
    for i in range(1, 4):
        highlighted_current_premise = get_words_at_indices(record["premise"], parse_indices(record[f"premise_highlighted_{i}"]))
        highlighted_premise_all.update(highlighted_current_premise)
        record[f"premise_highlighted_words_{i}"] = ",".join(highlighted_current_premise)
        highlighted_current_hypothesis = get_words_at_indices(record["hypothesis"], parse_indices(record[f"hypothesis_highlighted_{i}"]))
        highlighted_hypothesis_all.update(highlighted_current_hypothesis)
        record[f"hypothesis_highlighted_words_{i}"] = ",".join(highlighted_current_hypothesis)
    record["premise_highlighted_words_all"] = ",".join(highlighted_premise_all)
    record["hypothesis_highlighted_words_all"] = ",".join(highlighted_hypothesis_all)
    return record

def parse_indices(indices):
    if indices in ["{}", ""]:
        return []
    return [int(i) for i in indices.split(",")]

def get_words_at_indices(string, indices) -> list:
    split_string = string.split(" ")
    # Remove punctuation marks and empty strings
    return list(filter(lambda word: word != "", map(lambda i: re.sub(r"[.,!?]", "", split_string[i]), indices)))

def get_synonyms(word: str) -> set:
    synonyms = set()
    try:
        synsets = wordnet.synsets(word)
    except:
        return synonyms
    if synsets is None:
        return synonyms

    for synset in synsets:
        if synset is None:
            continue
        for lemma in synset.lemmas():
            if lemma is None:
                continue
            synonyms.add(lemma.name())
    return synonyms

def get_antonyms(word: str) -> set:
    antonyms = set()
    try:
        synsets = wordnet.synsets(word)
    except:
        return antonyms
    if synsets is None:
        return antonyms

    for synset in synsets:
        if synset is None:
            continue
        for lemma in synset.lemmas():
            if lemma is None:
                continue
            try:
                antonym_lemmas = lemma.antonyms()
            except:
                continue
            if antonym_lemmas:
                antonyms.add(antonym_lemmas[0].name())
    return antonyms

def get_hypernyms(word: str) -> set:
    hypernyms = set()
    try:
        synsets = wordnet.synsets(word)
    except:
        return hypernyms
    if synsets is None:
        return hypernyms

    for synset in synsets:
        if synset is None:
            continue
        for hypernym_synset in synset.hypernyms():
            hypernyms.update(hypernym_synset.lemma_names())
    return hypernyms

def get_hyponyms(word: str) -> set:
    hyponyms = set()
    try:
        synsets = wordnet.synsets(word)
    except:
        return hyponyms
    if synsets is None:
        return hyponyms

    for synset in synsets:
        if synset is None:
            continue
        for hyponym_synset in synset.hyponyms():
            hyponyms.update(hyponym_synset.lemma_names())
    return hyponyms

def add_simple_relation_column(record: dict, relation: str) -> dict:
    important_premise = set(record["premise_highlighted_words_all"].split(","))
    important_hypothesis = set(record["hypothesis_highlighted_words_all"].split(","))
    relation_pairs = set()
    for word_premise in important_premise:
        related_words = simple_relation_functions[relation](word_premise)
        for present_related_word in related_words.intersection(important_hypothesis):
            relation_pairs.add((word_premise, present_related_word))
    return { relation: len(relation_pairs) }

def are_co_hyponym(word_1: str, word_2: str) -> bool:
    common_hypernyms = get_hypernyms(word_1).intersection(get_hypernyms(word_2))
    return len(common_hypernyms) > 0

def add_co_hyponym_column(record: dict) -> dict:
    important_premise = set(record["premise_highlighted_words_all"].split(","))
    important_hypothesis = set(record["hypothesis_highlighted_words_all"].split(","))
    relation_pairs = set()
    for word_premise in important_premise:
        for word_hypothesis in important_hypothesis:
            if are_co_hyponym(word_premise, word_hypothesis):
                relation_pairs.add((word_premise, word_hypothesis))
    return { "co_hyponym": len(relation_pairs) }

def split_consecutive_sublists(indices: list) -> list:
    if len(indices) == 0:
        return []
    indices.sort()
    sublists = []
    current_sublist = [ indices[0] ]
    for item in indices[1:]:
        if current_sublist[-1] == item-1:
            current_sublist.append(item)
        else:
            sublists.append(current_sublist)
            current_sublist = [ item ]
    return sublists

def get_important_words_grouped(record: dict) -> set:
    words_and_groups = set()
    for sentence in ["premise", "hypothesis"]:
        for i in range(1, 4):
            word_indices_premise = parse_indices(record[f"{sentence}_highlighted_{i}"])
            grouped_word_indices_premise = split_consecutive_sublists(word_indices_premise)
            for word_group_indices in grouped_word_indices_premise:
                word_group = get_words_at_indices(record[sentence], word_group_indices)
                words_and_groups.update(word_group)
                words_and_groups.update(" ".join(word_group))
    return words_and_groups

def add_quantifier_column(record: dict) -> dict:
    important_words_grouped = get_important_words_grouped(record)
    quantifiers = list(filter(lambda phrase: phrase in all_quantifiers, important_words_grouped))
    return { "quantifiers": len(quantifiers) }

def get_important_words(record: dict) -> set:
    important_words = set(record["premise_highlighted_words_all"].split(","))
    important_words.update(record["hypothesis_highlighted_words_all"].split(","))
    return important_words

def add_numerical_column(record: dict) -> dict:
    tagged_tokens = []
    important_words = get_important_words(record)
    for sentence in [record["premise"], record["hypothesis"]]:
        tokens = word_tokenize(sentence)
        tagged_tokens.extend(pos_tag(tokens))
    important_numerical_words = list(filter(lambda tagged_token: tagged_token[1] == "CD" and tagged_token[0] in important_words, tagged_tokens))
    return { "numericals": len(important_numerical_words) }

simple_relation_functions = {
    "synonym": get_synonyms,
    "antonym": get_antonyms,
    "hypernym": get_hypernyms,
    "hyponym": get_hyponyms,
}

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Split dataset"
    )
    parser.add_argument('-o', '--output_path', required=True)
    args = parser.parse_args()

    esnli = esnli.map(transform_highlighted, num_proc=num_cpus)

    for key in simple_relation_functions.keys():
        print(f"Adding {key} column...")
        esnli = esnli.map(add_simple_relation_column, fn_kwargs={ "relation": key }, num_proc=num_cpus)

    print(f"Adding co-hyponym column...")
    esnli = esnli.map(add_co_hyponym_column, num_proc=num_cpus)

    print(f"Adding quantifier column...")
    esnli = esnli.map(add_quantifier_column, num_proc=num_cpus)

    print(f"Adding numerical column...")
    esnli = esnli.map(add_numerical_column, num_proc=num_cpus)

    esnli.save_to_disk(dataset_dict_path=args.output_path)
