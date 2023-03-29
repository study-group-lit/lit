from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from ferret import Benchmark
from multiprocessing import cpu_count

esnli = load_dataset("/workspace/students/lit/datasets/esnli_phenomena_all_words", split='validation')
model = RobertaForSequenceClassification.from_pretrained("/workspace/students/lit/models/roberta-base-finetuned-mnli")
tokenizer = RobertaTokenizer.from_pretrained("/workspace/students/lit/models/roberta-base-finetuned-mnli")

bench = Benchmark(model, tokenizer)

def explain(record, index):
    query = record["premise"] + "</s></s>" + record["hypothesis"]
    label = int(record["label"])
    explanation = bench.explain(query, target=label)
    explanation_html = bench.show_table(explanation).to_html()
    with open(f"./results/{index}.html", "w") as f:
        f.write(explanation_html)
    return record

esnli.filter(lambda r: r["quantifier"] > 0, num_proc=cpu_count()) \
    .select(range(9)) \
    .map(explain, num_proc=cpu_count, with_indices=True)


