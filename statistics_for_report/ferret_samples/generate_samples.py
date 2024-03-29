# README
# Filters e-SNLI validation split for samples containing quantifiers.
# Generates explanations and evaluations using ferret for the first 44 samples.
# The samples are stored in ./results

from datasets import DatasetDict
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from ferret import Benchmark
from multiprocessing import cpu_count

esnli = DatasetDict.load_from_disk(
    "/mnt/semproj/sem_proj22/proj_05/data/datasets/esnli_phenomena"
)
model = RobertaForSequenceClassification.from_pretrained(
    "/mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli"
)
tokenizer = RobertaTokenizer.from_pretrained(
    "/mnt/semproj/sem_proj22/proj_05/data/models/roberta-base-finetuned-mnli"
)

bench = Benchmark(model, tokenizer)


def explain(record, index):
    query = record["premise"] + "</s></s>" + record["hypothesis"]
    label = int(record["label"])
    explanation = bench.explain(query, target=label)
    explanation_html = bench.show_table(explanation).to_html()
    with open(f"./results/{index}_explanation.html", "w") as f:
        f.write(explanation_html)

    evaluation = bench.evaluate_explanations(explanation, target=label)
    evalutation_html = bench.show_evaluation_table(evaluation).to_html()
    with open(f"./results/{index}_evaluation.html", "w") as f:
        f.write(evalutation_html)

    return record


esnli["validation"].filter(lambda r: r["quantifiers"] > 0, num_proc=cpu_count()).select(
    range(0, 44)
).map(explain, with_indices=True)
