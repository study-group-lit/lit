from datasets import load_from_disk, concatenate_datasets, Dataset
import pandas as pd
from statistics import mean, median
import json

esnli_phenomena = load_from_disk("/workspace/students/lit/datasets/esnli_phenomena")["validation"].to_pandas()
esnli_evaluations = load_from_disk("/workspace/students/lit/datasets/esnli_evaluations_42/esnli_evaluations_chunk_0")
for i in range(1, 5):
    esnli_evaluations_chunk = load_from_disk(f"/workspace/students/lit/datasets/esnli_evaluations_42/esnli_evaluations_chunk_{i}")
    esnli_evaluations = concatenate_datasets([esnli_evaluations, esnli_evaluations_chunk])
esnli_evaluations = esnli_evaluations.to_pandas()

esnli_evaluations_hypothesis_only = load_from_disk("/workspace/students/lit/datasets/esnli_evaluations_hypothesis_only_42").to_pandas()

def merge(record):
    candidate_partners = esnli_evaluations.filter(lambda r: r["premise"] == record["premise"] and r["hypothesis"] == record["hypothesis"])
    if candidate_partners.num_rows > 1:
        print(f"Warning: Duplicate premise hypothesis pair for record: {record}")
    if candidate_partners.num_rows == 0:
        raise RuntimeError(f"Missing premise hypothesis pair for record: {record}")
    return {**record, **candidate_partners[0]}

esnli_evaluations_phenomena_pd = pd.merge(esnli_evaluations, esnli_phenomena, on=["premise", "hypothesis"], suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

esnli_evaluations_phenomena = Dataset.from_pandas(esnli_evaluations_phenomena_pd)
esnli_evaluations_phenomena.save_to_disk("/workspace/students/lit/datasets/esnli_evaluations_42_phenomena")

esnli_evaluations_hypothesis_only_phenomena_pd = pd.merge(esnli_evaluations_hypothesis_only, esnli_phenomena, on=["premise", "hypothesis"], suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
esnli_evaluations_hypothesis_only_phenomena = Dataset.from_pandas(esnli_evaluations_hypothesis_only_phenomena_pd)
esnli_evaluations_hypothesis_only_phenomena.save_to_disk("/workspace/students/lit/datasets/esnli_evaluations_hypothesis_only_42_phenomena")


def calculate_score(dataset, row_value_selector, score):
    # Use python map, as we want values, not table entries
    value_dataset = map(row_value_selector, dataset)
    # Score takes an iter and returns a value
    return score(value_dataset)

def calculate_dataset_scores(dataset):
    scores = [sum, min, max, median, mean]
    score_names = ["sum", "min", "max", "median", "mean"]
    explainer_names = ["shap", "lime", "integrated_gradient", "integrated_gradient_multiply_by_inputs"]
    evaluator_names = ["comprehensiveness", "sufficiency", "loo", "auprc_plausibility", "tokenf1_plausibility", "tokeniou_plausibility"]
    dataset_scores = {}
    for i, explainer_name in zip(range(4), explainer_names): # 4 explainers
        dataset_scores[explainer_name] = {}
        for j, evaluator_name in zip(range(6), evaluator_names): # 6 evaluators for each explainers
            row_value_selector = lambda row: row["evaluations"][i][j]
            evaluation_scores = [calculate_score(dataset, row_value_selector, score) for score in scores]
            named_evaluation_scores = {name: value for (name, value) in zip(score_names, evaluation_scores)}
            dataset_scores[explainer_name][evaluator_name] = named_evaluation_scores
    return dataset_scores

esnli_evaluations_hypothesis_only_phenomena_scores = calculate_dataset_scores(esnli_evaluations_hypothesis_only_phenomena)
with open(f"/workspace/students/lit/datasets/esnli_evaluations_hypothesis_only_phenomena_scores.json", "w+") as f:
    f.write(json.dumps(esnli_evaluations_hypothesis_only_phenomena_scores))

esnli_evaluations_phenomena_scores = calculate_dataset_scores(esnli_evaluations_phenomena)
with open(f"/workspace/students/lit/datasets/esnli_evaluations_phenomena_scores.json", "w+") as f:
    f.write(json.dumps(esnli_evaluations_phenomena_scores))
