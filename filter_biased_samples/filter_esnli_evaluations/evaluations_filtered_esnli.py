"""
In contrast to evaluations_esnli.py, this script does not need to calculate the evaluations, as they are already a part of the dataset.
This script only needs to filter esnli (validations) and then calculate the metrics on the result.
"""
from datasets import load_from_disk
from statistics import mean, median
import json

esnli = load_from_disk("/mnt/semproj/sem_proj22/proj_05/data/datasets/esnli_phenomena_with_predictions")


# The item ferret_explanations contains the scores for the explanations of Shap, LIME, Integrated Gradient and Integrated Gradients multiplying inputs
# The item evaluations contains scores for each explanation for AOPC comprehensiveness, AOPC sufficiency, Tau LOO, AUPRC, Token F1, Token IOU
# https://github.com/g8a9/ferret/blob/b1343501db6367ca9048283862f8e0763c72e4ba/ferret/benchmark.py#L88

def filter_predicted_samples(records):
    predicted_label = records["prediction"]
    label = records["label"]
    return predicted_label == label

def calculate_score(dataset, row_value_selector, score):
    # Use python map, as we want values, not table entries
    value_dataset = map(row_value_selector, dataset)
    # Score takes an iter and returns a value
    return score(value_dataset)

def calculate_dataset_scores(dataset):
    scores = [min, max, median, mean]
    score_names = ["min", "max", "median", "mean"]
    explainer_names = ["shap", "lime", "integrated_gradient", "integrated_gradient_multiply_by_inputs"]
    evaluator_names = ["comprehensiveness", "sufficiency", "loo", "auprc_plausibility", "tokenf1_plausibility", "tokeniou_plausibility"]
    dataset_scores = {}
    for i, explainer_name in enumerate(explainer_names): # 4 explainers
        dataset_scores[explainer_name] = {}
        for j, evaluator_name in enumerate(evaluator_names): # 6 evaluators for each explainers
            row_value_selector = lambda row: row["evaluations"][i][j]
            evaluation_scores = [calculate_score(dataset, row_value_selector, score) for score in scores]
            named_evaluation_scores = {name: value for (name, value) in zip(score_names, evaluation_scores)}
            dataset_scores[explainer_name][evaluator_name] = named_evaluation_scores
    return dataset_scores


filtered_esnli = esnli.filter(filter_predicted_samples)
print("Calculating scores")
filtered_esnli_scores = calculate_dataset_scores(filtered_esnli)
filtered_esnli.save_to_disk("/mnt/semproj/sem_proj22/proj_05/data/datasets/esnli_phenomena_with_predictions_filtered")
with open("/mnt/semproj/sem_proj22/proj_05/data/datasets/esnli_phenomena_with_predictions_filtered.json", "w+") as f:
    f.write(json.dumps(filtered_esnli_scores))