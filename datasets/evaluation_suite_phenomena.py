import evaluate
from evaluate.evaluation_suite import SubTask
from multiprocessing import cpu_count
from datasets import DatasetDict


# code from https://discuss.huggingface.co/t/combining-metrics-for-multiclass-predictions-evaluations/21792/11
class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs
    
    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)
    
    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()


class Suite(evaluate.EvaluationSuite):
    def __init__(self, name):
        super().__init__(name)
        self.esnli_phenomena = DatasetDict.load_from_disk("/workspace/students/lit/datasets/esnli_phenomena")
        self.phenomena = ["synonym", "antonym", "hypernym", "hyponym", "co_hyponym", "quantifiers", "numericals"]
        self.suite = list(map(lambda phenomenon: self.task_for(phenomenon), self.phenomena))
    
    def task_for(self, phenomenon: str):
        metric = evaluate.combine([
            evaluate.load("accuracy"),
            evaluate.load("matthews_correlation"),
            ConfiguredMetric(evaluate.load("f1"), average="macro"),
            evaluate.load("BucketHeadP65/confusion_matrix")
        ])

        print(f"Filtering data by phenomenon {phenomenon}...")
        data = self.esnli_phenomena["validation"].filter(lambda r: r[phenomenon] > 0, num_proc=cpu_count())
        data.info.description = phenomenon
        print(f"Dataset for {phenomenon} has size {data.num_rows}...")

        return SubTask(
                task_type="text-classification",
                data=data,
                args_for_task={
                    "metric": metric,
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "label_column": "label",
                    "label_mapping": {
                        "ENTAILMENT": 0,
                        "NEUTRAL": 1,
                        "CONTRADICTION": 2
                    }
                }
            )