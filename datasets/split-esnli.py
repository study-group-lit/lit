from datasets import load_dataset
import re
import nltk
from nltk.corpus import wordnet
import multiprocessing

nltk.download('wordnet')
esnli = load_dataset("../datasets/esnli.py")

def transform_highlighted(record: dict) -> dict:
    highlighted_premise_all = set()
    highlighted_hypothesis_all = set()
    for i in range(1, 4):
        highlighted_current_premise = get_words_at_indices(record["premise"], parse_indices(record[f"premise_highlighted_{i}"]))
        highlighted_premise_all.update(highlighted_current_premise)
        record[f"premise_highlighted_{i}"] = ",".join(highlighted_current_premise)
        highlighted_current_hypothesis = get_words_at_indices(record["hypothesis"], parse_indices(record[f"hypothesis_highlighted_{i}"]))
        highlighted_hypothesis_all.update(highlighted_current_hypothesis)
        record[f"hypothesis_highlighted_{i}"] = ",".join(highlighted_current_hypothesis)
    record["highlighted_premise_all"] = ",".join(highlighted_premise_all)
    record["highlighted_hypothesis_all"] = ",".join(highlighted_hypothesis_all)
    return record

def parse_indices(indices: str) -> list[int]:
    if indices in ["{}", ""]:
        return []
    return [int(i) for i in indices.split(",")]

def get_words_at_indices(string: str, indices: list[int]) -> str:
    split_string = string.split(" ")
    # Remove punctuation marks and empty strings
    return filter(lambda word: word != "", map(lambda i: re.sub(r"[.,!?]", "", split_string[i]), indices))

num_cpus = multiprocessing.cpu_count()
splits = ["train", "test", "validation"]
for split in splits:
    esnli[split] = esnli[split].map(transform_highlighted, num_proc=num_cpus)


def get_synonyms(word: str) -> set[str]:
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def get_antonyms(word: str) -> set[str]:
    antonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    return antonyms

def get_hypernyms(word: str) -> set[str]:
    hypernyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.hypernyms():
                hypernyms.add(lemma.hypernyms()[0].name())
    return hypernyms

def get_hyponyms(word: str) -> set[str]:
    hyponyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.hyponyms():
                hyponyms.add(lemma.hyponyms()[0].name())
    return hyponyms

simple_relation_functions = {
    "synonym": get_synonyms,
    "antonym": get_antonyms,
    "hypernym": get_hypernyms,
    "hyponym": get_hyponyms,
}

def add_simple_relation_column(record: dict, relation: str) -> dict:
    important_premise = set(record["highlighted_premise_all"].split(","))
    important_hypothesis = set(record["highlighted_hypothesis_all"].split(","))
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
    important_premise = set(record["highlighted_premise_all"].split(","))
    important_hypothesis = set(record["highlighted_hypothesis_all"].split(","))
    relation_pairs = set()
    for word_premise in important_premise:
        for word_hypothesis in important_hypothesis:
            if are_co_hyponym(word_premise, word_hypothesis):
                relation_pairs.add((word_premise, word_hypothesis))
    return { "co_hyponym": len(relation_pairs) }

splits = ["train", "test", "validation"]
for split in splits:
    for key in simple_relation_functions.keys():
        print(f"Adding {key} column to {split} split...")
        esnli[split] = esnli[split].map(add_simple_relation_column, fn_kwargs={ "relation": key }, num_proc=num_cpus)
    print(f"Adding co-hyponym column to {split} split...")
    esnli[split] = esnli[split].map(add_co_hyponym_column, num_proc=num_cpus)
    esnli[split].to_csv(f"../datasets/esnli_{split}_phenomena.csv")
