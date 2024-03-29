# README
# This file contains a script to train a roberta-base model only on the hypotheses 
# from the MultiNLI dataset for three epochs.

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score
from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer
import os
import argparse

parser = argparse.ArgumentParser(
                    prog = 'Hypothesis only MNLI training',
                    description = 'Trains RoBERTa using only MNLI hypothesis')
parser.add_argument("seed", default=42, type=int, nargs='?')
args = parser.parse_args()

seed = args.seed
model_path = "../../models/roberta-base-mnli-hypothesis-only/" + str(seed)

if not os.path.exists(model_path):
    os.makedirs(model_path)
checkpoint_path = os.path.join(model_path, "checkpoints")
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

mnli = load_dataset("multi_nli")

roberta = RobertaModel.from_pretrained("roberta-base")

config = RobertaConfig.from_json_file("../../models/sequence_classification.json")
model = RobertaForSequenceClassification(config)

state_dict = roberta.state_dict()
del state_dict["pooler.dense.weight"]
del state_dict["pooler.dense.bias"]
model.roberta.load_state_dict(state_dict)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def map_mnli_split(split, tokenizer):
    split = split.map(lambda d: {"x": d["hypothesis"]}, batched=True)
    split = split.map(lambda d: tokenizer(d["x"], padding="max_length", truncation=True), batched=True)
    split.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return split

mnli_train = mnli["train"]
# Only use hypothesis
mnli_train = map_mnli_split(mnli_train, tokenizer)

mnli_val = mnli["validation_matched"]
mnli_val = map_mnli_split(mnli_val, tokenizer)

def compute_metrics(pred):
    """
    Shows a few helpful metrics and saves them in specified directory
    :param pred: list
    """

    true = pred.label_ids
    predicted = pred.predictions.argmax(-1)

    return {
        "MCC": matthews_corrcoef(true, predicted),
        "F1": f1_score(true, predicted, average='macro'),
        "Acc": accuracy_score(true, predicted),
        "BAcc": balanced_accuracy_score(true, predicted),
    }

training_args = TrainingArguments(
    output_dir=checkpoint_path,          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    gradient_accumulation_steps=2,      # 2 accumulation steps, as we can maximally use 8 as batch size
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.0,               # strength of weight decay
    learning_rate=2e-5,
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=10000,               # log & save weights each logging_steps
    save_steps=10000,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
    log_level="info",
    seed=seed
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=mnli_train,         # training dataset
    eval_dataset=mnli_val,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)