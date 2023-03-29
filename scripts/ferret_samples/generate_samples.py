from datasets import DatasetDict
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from ferret import Benchmark
from multiprocessing import cpu_count

esnli = DatasetDict.load_from_disk("/workspace/students/lit/datasets/esnli_phenomena")
model = RobertaForSequenceClassification.from_pretrained("/workspace/students/lit/models/roberta-base-finetuned-mnli")
tokenizer = RobertaTokenizer.from_pretrained("/workspace/students/lit/models/roberta-base-finetuned-mnli")

bench = Benchmark(model, tokenizer)

def explain(record, index):
    query = record["premise"] + "</s></s>" + record["hypothesis"]
    label = int(record["label"])
    explanation = bench.explain(query, target=label)
    explanation_html = bench.show_table(explanation).to_html()
    with open(f"~/lit/scripts/ferret_samples/results/{index}_explanation.html", "w") as f:
        f.write(explanation_html)

    evaluation = bench.evaluate_explanations(explanation, target=label)
    evalutation_html = bench.show_evaluation_table(evaluation).to_html()
    with open(f"~/lit/scripts/ferret_samples/results/{index}_evaluation.html" , "w" ) as f:
        f.write(evalutation_html)

    return record

esnli["validation"].filter(lambda r: r["quantifiers"] > 0, num_proc=cpu_count()) \
    .select(range(0, 44)) \
    .map(explain, with_indices=True)


