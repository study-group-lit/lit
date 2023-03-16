from datasets import load_from_disk, concatenate_datasets, Dataset
import pandas as pd

esnli_phenomena = load_from_disk("../../lit-data/datasets/esnli_phenomena")["validation"].to_pandas()
esnli_evaluations = load_from_disk("../../lit-data/datasets/esnli_evaluations_42/esnli_evaluations_chunk_0")
for i in range(1, 5):
    esnli_evaluations_chunk = load_from_disk(f"../../lit-data/datasets/esnli_evaluations_42/esnli_evaluations_chunk_{i}")
    esnli_evaluations = concatenate_datasets([esnli_evaluations, esnli_evaluations_chunk])
esnli_evaluations = esnli_evaluations.to_pandas()

esnli_evaluations_hypothesis_only = load_from_disk("../../lit-data/datasets/esnli_evaluations_hypothesis_only_42").to_pandas()

def merge(record):
    candidate_partners = esnli_evaluations.filter(lambda r: r["premise"] == record["premise"] and r["hypothesis"] == record["hypothesis"])
    if candidate_partners.num_rows > 1:
        print(f"Warning: Duplicate premise hypothesis pair for record: {record}")
    if candidate_partners.num_rows == 0:
        raise RuntimeError(f"Missing premise hypothesis pair for record: {record}")
    return {**record, **candidate_partners[0]}

esnli_evaluations_phenomena_pd = pd.merge(esnli_evaluations, esnli_phenomena, on=["premise", "hypothesis"], suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

esnli_evaluations_phenomena = Dataset.from_pandas(esnli_evaluations_phenomena_pd)
esnli_evaluations_phenomena.save_to_disk("../../lit-data/datasets/esnli_evaluations_42_phenomena")

esnli_evaluations_hypothesis_only_phenomena_pd = pd.merge(esnli_evaluations_hypothesis_only, esnli_phenomena, on=["premise", "hypothesis"], suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
esnli_evaluations_hypothesis_only_phenomena = Dataset.from_pandas(esnli_evaluations_hypothesis_only_phenomena_pd)
esnli_evaluations_hypothesis_only_phenomena.save_to_disk("../../lit-data/datasets/esnli_evaluations_hypothesis_only_42_phenomena")

