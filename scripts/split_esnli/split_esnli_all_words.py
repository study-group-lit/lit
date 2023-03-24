from datasets import load_dataset
import re
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, ngrams
import multiprocessing

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

def split_sentence(sentence: dict) -> set:
    split = set(sentence.split(" "))
    # Remove punctuation marks and empty strings
    return set(filter(lambda word: word != "", map(lambda word: re.sub(r"[.,!?]", "", word), split)))

def get_synonyms(word: str) -> set:
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def get_antonyms(word: str) -> set:
    antonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())
    return antonyms

def get_hypernyms(word: str) -> set:
    hypernyms = set()
    for synset in wordnet.synsets(word):
        for hypernym_synset in synset.hypernyms():
            hypernyms.update(hypernym_synset.lemma_names())
    return hypernyms

def get_hyponyms(word: str) -> set:
    hyponyms = set()
    for synset in wordnet.synsets(word):
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
    shingles = []
    for sentence in [record["premise"], record["hypothesis"]]:
        tokens = word_tokenize(sentence)
        for n in range(1, 5):
            shingles.extend(ngrams(tokens, n))
    shingles = map(lambda shingle: " ".join(shingle), shingles)
    quantifiers = list(filter(lambda shingle: shingle in all_quantifiers, shingles))
    return { "quantifiers": len(quantifiers) }

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

    esnli.save_to_disk(dataset_dict_path="../../../lit-data/datasets/esnli_phenomena_all_words")