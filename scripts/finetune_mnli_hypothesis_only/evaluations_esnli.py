from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer
from ferret import Benchmark
from ferret import SHAPExplainer, LIMEExplainer, IntegratedGradientExplainer
from statistics import mean, median
import json
import argparse
import os

parser = argparse.ArgumentParser(
                    prog = 'Hypothesis only esnli evaluation calculation',
                    description = 'Calculate evaluations on esnli data on the hypothesis only RoBERTa model')
parser.add_argument("-m", "--model", default="../../models/roberta-base-mnli-hypothesis-only/42/", type=str)
parser.add_argument("seed", default=42, type=int, nargs='?')
args = parser.parse_args()

model_path = args.model
seed = args.seed

if not os.path.exists(model_path):
    raise IOError("Model path does not exist")

esnli = load_dataset("../../datasets/esnli.py", split='validation')
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


explainers = [
    SHAPExplainer(model, tokenizer),
    LIMEExplainer(model, tokenizer),
    IntegratedGradientExplainer(model, tokenizer, multiply_by_inputs=False),
    IntegratedGradientExplainer(model, tokenizer, multiply_by_inputs=True)
]
bench = Benchmark(model, tokenizer, explainers=explainers)


BPE_DIVIDER_TOKEN = "Ġ"
def create_token_rationale_encoding(hypothesis, esnli_explanation):
    """
    Creates a list, in which for each token is specified, if the token is part of the explanation (1), or not (0).
    https://github.com/g8a9/ferret/blob/55978fa7528e0cf2f88ea767288f5d3048dd7553/ferret/benchmark.py#L158
    Ferret calls this a one-hot-encoding, though the returned list may contain several 1's 
    """
    tokens = tokenizer.tokenize(hypothesis)
    one_hot_encoding = []
    current_word_index = 0
    for token in tokens:
        if token.startswith(BPE_DIVIDER_TOKEN):
            current_word_index+=1
        token_code = 1 if str(current_word_index) in esnli_explanation else 0
        one_hot_encoding.append(token_code)
    return one_hot_encoding


def calculate_evaluations(row):
    hypothesis = row["hypothesis"]
    label = row["label"]
    esnli_explanation = row["sentence2_highlighted_1"]
    rationale = create_token_rationale_encoding(hypothesis, esnli_explanation)
    ferret_explanations = bench.explain(hypothesis, target=label)
    
    row["ferret_explanations"] = [explanation.scores for explanation in ferret_explanations]
    evaluations = [bench.evaluate_explanation(explanation, label, rationale) for explanation in ferret_explanations]
    row["evaluations"] = [[evaluation_score.score for evaluation_score in evaluation.evaluation_scores] for evaluation in evaluations]
    return row

# The item ferret_explanations contains the scores for the explanations of Shap, LIME, Integrated Gradient and Integrated Gradients multiplying inputs
# The item evaluations contains scores for each explanation for AOPC comprehensiveness, AOPC sufficiency, Tau LOO, AUPRC, Token F1, Token IOU
# https://github.com/g8a9/ferret/blob/b1343501db6367ca9048283862f8e0763c72e4ba/ferret/benchmark.py#L88


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


esnli_with_evaluations = esnli.map(calculate_evaluations)
dataset_scores = calculate_dataset_scores(esnli_with_evaluations)

esnli_with_evaluations.save_to_disk(f"../../datasets/esnli_evaluations_hypothesis_only_{seed}.hf")
with open(f"../../datasets/esnli_evaluation_scores_hypothesis_only_{seed}.json", "w+") as f:
    f.write(json.dumps(dataset_scores))


