import os
from datasets import load_from_disk, load_dataset, concatenate_datasets, ClassLabel, Value
from transformers import TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score
from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer
import multiprocessing
num_cpus = multiprocessing.cpu_count()
import datasets
datasets.disable_caching()

# check paths and create if needed
model_path = "/workspace/students/lit/models/roberta-base-finetuned-recast/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
checkpoint_path = os.path.join(model_path, "checkpoints")
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

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
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def adapt_mnli_schema(mnli):
    return mnli.map(lambda record: {
        "premise": record["premise"],
        "hypothesis": record["hypothesis"],
        "label": record["label"]
    }, num_proc=num_cpus)

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


# build the classification model from the pretrained language model
# and an untrained language model
roberta = RobertaModel.from_pretrained("roberta-base")

# config is taken from official fine-tuning of roberta for mnli
# only change are the label keys that were changed from LABEL_0, LABEL_1, LABEL_2
# to ENTAILMENT, NEUTRAL and HYPOTHESIS for better readability
config = RobertaConfig.from_json_file("../../models/sequence_classification.json")
# initialize untrained roberta sequence classifier
model = RobertaForSequenceClassification(config)

# get model weights of PLM
state_dict = roberta.state_dict()
# remove unneeded pooler weights. If not done, loading the PLM will crash
# pooling will be handled by the classification head
del state_dict["pooler.dense.weight"]
del state_dict["pooler.dense.bias"]
# load PLM into untrained sequence classifier
model.roberta.load_state_dict(state_dict)

# load and preprocess datasets
cnn = load_from_disk("/workspace/students/lit/datasets/cnn_dataset_augmented")
dailymail = load_from_disk("/workspace/students/lit/datasets/dailymail_dataset_augmented")
mnli = load_dataset("multi_nli")

def encode_label(recast_dataset):
    """
    Currently label is of type string but contains the encoded labels as integers
    """
    return recast_dataset.cast_column("label", Value("int64")).cast_column("label", ClassLabel(names= [
        "entailment",
        "neutral",
        "contradiction"
      ]))

recast_train_unprocessed = concatenate_datasets([
    adapt_mnli_schema(mnli["train"]),
    encode_label(cnn),
    encode_label(dailymail)
])

recast_train = preprocess_dataset(recast_train_unprocessed)
recast_val = preprocess_dataset(mnli["validation_matched"])

# training-relevant params from https://huggingface.co/textattack/roberta-base-MNLI/blob/main/training_args.bin
# performance-relevant params obtained by hand-tuning
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
    train_dataset=recast_train,           # training dataset
    eval_dataset=recast_val,              # evaluation dataset
    compute_metrics=compute_metrics,    # the callback that computes metrics of interest
)

# train, evaluate and save everything
trainer.train(resume_from_checkpoint=False)
trainer.evaluate()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
