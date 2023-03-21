from datasets import DatasetDict
from multiprocessing import cpu_count

esnli_phenomena = DatasetDict.load_from_disk("../../lit-data/datasets/esnli_phenomena_all_words")
phenomena = ["synonym", "antonym", "hypernym", "hyponym", "co_hyponym", "quantifiers", "numericals"]
splits = ["train", "test", "validation"]
for split in splits:
    for phenomenon in phenomena:
        filtered = esnli_phenomena[split].filter(lambda r: r[phenomenon] > 0, num_proc=cpu_count())
        print(f"{split} {phenomenon}: {filtered.num_rows}")