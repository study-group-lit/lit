from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer
from ferret import Benchmark
from ferret import SHAPExplainer, LIMEExplainer, IntegratedGradientExplainer
from statistics import mean, median
import json
import argparse
from math import ceil

parser = argparse.ArgumentParser(
                    prog = 'Esnli evaluation calculation',
                    description = 'Calculate evaluations on esnli data on the RoBERTa model')
parser.add_argument("-v", "--variant", type=str, default="1")
# For each row the SHAP explainer alone takes 16s. Running all tasks would take ~150h
# Thus we chunk the task into pieces of 2000 items (esnli validation has about 10000)
parser.add_argument("-c", "--chunk_index", type=int, default=0)
args = parser.parse_args()

variant = args.variant
chunk_index = args.chunk_index

CHUNK_SIZE = 2000

esnli = load_dataset("../../datasets/esnli.py", split='validation')
esnli_size = len(esnli)
maximum_chunk = ceil(esnli_size / CHUNK_SIZE)

esnli_start_index = chunk_index*CHUNK_SIZE
if esnli_start_index >= esnli_size:
    print("Choose a smaller chunk index")
    exit(1)

esnli_end_index = min((chunk_index+1)*CHUNK_SIZE, esnli_size)
esnli = esnli.select(range(esnli_start_index, esnli_end_index))

model_path = f"/workspace/students/lit/models/roberta-base-finetuned-mnli/"
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
def create_one_hot(hypothesis, esnli_explanation):
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
    premise = row["premise"]
    hypothesis = row["hypothesis"]
    query = premise + "</s></s>" + hypothesis

    label = row["label"]
    esnli_premise_explanation = row["sentence1_highlighted_1"]
    esnli_hypotheses_explanation = row["sentence2_highlighted_1"]
    premise_rationale = create_one_hot(premise, esnli_premise_explanation)
    hypotheses_rationale = create_one_hot(hypothesis, esnli_hypotheses_explanation)
    rationale = premise_rationale + [0, 0] + hypotheses_rationale

    ferret_explanations = bench.explain(query, target=label)
    
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


esnli_with_evaluations.save_to_disk(f"../../datasets/esnli_evaluations_{variant}_chunk_{chunk_index}.hf")
with open(f"../../datasets/esnli_evaluation_scores_{variant}_chunk_{chunk_index}.json", "w+") as f:
    f.write(json.dumps(dataset_scores))