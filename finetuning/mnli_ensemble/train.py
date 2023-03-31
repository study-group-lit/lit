# README
# This file contains a script to ensemble and train a model as described in section 2.4 of the report.

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score
from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer

import os
import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

model_path = "/mnt/semproj/sem_proj22/proj_05/data/models/roberta-ensemble-finetuned-mnli/"
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
checkpoint_path = os.path.join(model_path, "checkpoints")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)

class EnsembleForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.bias_head = nn.Linear(config.hidden_size, 1)

        self.hypothesis_only = RobertaForSequenceClassification(config)
        self.hypothesis_only.eval()

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        hypothesis_input_ids = None,
        hypothesis_attention_mask = None,
        hypothesis_token_type_ids = None,
        hypothesis_position_ids = None,
        hypothesis_head_mask = None,
        hypothesis_inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        # copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/roberta/modeling_roberta.py#L1190
        # new: lines 77-95, 120-121
        # add the hypothesis-only inference and bias entropy
        # as described in https://aclanthology.org/D19-1418.pdf
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        entropy = None
        if self.training:
            bias_weight = F.softplus(self.bias_head(sequence_output[:, 0, :])) # take only <s> token

            with torch.no_grad():
                hypothesis_logits = self.hypothesis_only(
                    hypothesis_input_ids,
                    attention_mask=hypothesis_attention_mask,
                    token_type_ids=hypothesis_token_type_ids,
                    position_ids=hypothesis_position_ids,
                    head_mask=hypothesis_head_mask,
                    inputs_embeds=hypothesis_inputs_embeds,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,        
                ).logits
            bias = bias_weight*hypothesis_logits
            probs = F.softmax(bias, dim=-1)
            entropy = torch.mean(torch.sum(-probs*torch.log(probs), dim=-1))
            logits += bias

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            
            if entropy is not None:
                loss += 0.03*entropy
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

config = RobertaConfig.from_json_file("../../models/sequence_classification.json")
model = EnsembleForSequenceClassification(config)

roberta = RobertaModel.from_pretrained("roberta-base")
state_dict = roberta.state_dict()
del state_dict["pooler.dense.weight"]
del state_dict["pooler.dense.bias"]
model.roberta.load_state_dict(state_dict)

hypothesis_only = RobertaForSequenceClassification.from_pretrained("/mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli-hypothesis-only/1337/")
model.hypothesis_only.load_state_dict(hypothesis_only.state_dict())

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def preprocess_dataset(dataset, premise_column="premise", hypothesis_column="hypothesis"):
    """
    tokenizes columns with premise and hypothesis of a dataset
    """
    dataset = dataset.map(lambda d: tokenizer(
        text=d[premise_column],
        text_pair=d[hypothesis_column],
        padding="max_length",
        truncation=True
        ), batched=True)
    def tokenize_hypothesis_only(d):
        temp = tokenizer(
            text=d[premise_column],
            text_pair=d[hypothesis_column],
            padding="max_length",
            truncation=True
        )
        return {
            "hypothesis_input_ids": temp["input_ids"],
            "hypothesis_attention_mask": temp["attention_mask"],
            }
    dataset = dataset.map(tokenize_hypothesis_only, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "hypothesis_input_ids", "hypothesis_attention_mask"])
    return dataset

def compute_metrics(pred):
    """
    Calculates a few helpful metrics
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

mnli = load_dataset("multi_nli")
mnli_train = preprocess_dataset(mnli["train"])
mnli_val = preprocess_dataset(mnli["validation_matched"])

training_args = TrainingArguments(
    output_dir=checkpoint_path,         # output directory
    num_train_epochs=3,                 # total number of training epochs
    per_device_train_batch_size=8,      # original training has batch_size 16 =>
    gradient_accumulation_steps=2,      # 2 accumulation steps, as we can maximally use 8 as batch size
    per_device_eval_batch_size=32,      # batch size for evaluation
    warmup_steps=0,                     # number of warmup steps for learning rate scheduler
    weight_decay=0.0,                   # strength of weight decay
    learning_rate=2e-5,                 # learning rate
    logging_dir='./logs',               # directory for storing logs
    load_best_model_at_end=True,        # load the best model when finished training
    logging_steps=10000,                # log & save weights each logging_steps
    save_steps=10000,
    evaluation_strategy="steps",        # evaluate each `logging_steps`
    log_level="info",                   # log evaluation results
    seed=1337,
)

trainer = Trainer(
    model=model,                        # the instantiated Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    train_dataset=mnli_train,           # training dataset
    eval_dataset=mnli_val,              # evaluation dataset
    compute_metrics=compute_metrics,    # the callback that computes metrics of interest
)

trainer.train(resume_from_checkpoint=False)
trainer.evaluate()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)