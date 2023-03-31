# README
# This file contains a script to detect the liguistic phenomena of synonym, antonym,
# hypernym, hyponym, co_hyponym, quantifiers and numerals in a dataset. For each
# phenomenon a column will be added to the dataset which contains a number. If the
# number is >0, the respective phenomenon has been found in the sample.

from typing import List
from attr import dataclass
from datasets import load_dataset
import re
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from multiprocessing import cpu_count
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore")

nltk.download('wordnet')
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

def contains_quantifier(sentence):
    tokens = pos_tag(word_tokenize(sentence))
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

def parse_indices(indices):
    if indices in ["{}", ""]:
        return []
    return [int(i) for i in indices.split(",")]

def get_words_at_indices(string, indices) -> list:
    split_string = string.split(" ")
    # Remove punctuation marks and empty strings
    return list(filter(lambda word: word != "", map(lambda i: re.sub(r"[.,!?]", "", split_string[i]), indices)))

def split_sentence(sentence: dict) -> set:
    split = set(sentence.split(" "))
    # Remove punctuation marks and empty strings
    return set(filter(lambda word: word != "", map(lambda word: re.sub(r"[.,!?]", "", word), split)))

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
    words_premise = split_sentence(record["premise"])
    words_hypothesis = split_sentence(record["hypothesis"])
    relation_pairs = set()
    for word_premise in words_premise:
        related_words = simple_relation_functions[relation](word_premise)
        for present_related_word in related_words.intersection(words_hypothesis):
            relation_pairs.add((word_premise, present_related_word))
    return { relation: len(relation_pairs) }

def are_co_hyponym(word_1: str, word_2: str) -> bool:
    common_hypernyms = get_hypernyms(word_1).intersection(get_hypernyms(word_2))
    return len(common_hypernyms) > 0

def add_co_hyponym_column(record: dict) -> dict:
    words_premise = split_sentence(record["premise"])
    words_hypothesis = split_sentence(record["hypothesis"])
    relation_pairs = set()
    for word_premise in words_premise:
        for word_hypothesis in words_hypothesis:
            if are_co_hyponym(word_premise, word_hypothesis):
                relation_pairs.add((word_premise, word_hypothesis))
    return { "co_hyponym": len(relation_pairs) }

def add_quantifier_column(record: dict) -> dict:
    return { "quantifiers": contains_quantifier(record["premise"]) or contains_quantifier(record["hypothesis"]) }

def add_numerical_column(record: dict) -> dict:
    tagged_tokens = []
    for sentence in [record["premise"], record["hypothesis"]]:
        tokens = word_tokenize(sentence)
        tagged_tokens.extend(pos_tag(tokens))
    important_numerical_words = list(filter(lambda tagged_token: tagged_token[1] == "CD", tagged_tokens))
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
    parser.add_argument('-d', '--dataset_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    args = parser.parse_args()

    num_cpus = cpu_count()
    dataset = load_dataset(args.dataset_path)

    for key in simple_relation_functions.keys():
        print(f"Adding {key} column...")
        dataset = dataset.map(add_simple_relation_column, fn_kwargs={ "relation": key }, num_proc=num_cpus)

    print(f"Adding co-hyponym column...")
    dataset = dataset.map(add_co_hyponym_column, num_proc=num_cpus)

    print(f"Adding quantifier column...")
    dataset = dataset.map(add_quantifier_column, num_proc=num_cpus)

    print(f"Adding numerical column...")
    dataset = dataset.map(add_numerical_column, num_proc=num_cpus)

    dataset.save_to_disk(dataset_dict_path=args.output_path)
