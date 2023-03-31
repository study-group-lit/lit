import evaluate
from evaluate.evaluation_suite import SubTask

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

        self.suite = [
            self.task_for("sick", "test", "sentence_A", "sentence_B")
        ]
    
    def task_for(self, data: str, split: str, premise: str = "premise", hypothesis: str = "hypothesis"):
        metric = evaluate.combine([
            evaluate.load("accuracy"),
            evaluate.load("matthews_correlation"),
            ConfiguredMetric(evaluate.load("f1"), average="macro"),
        ])

        return SubTask(
                task_type="text-classification",
                data=data,
                split=split,
                subset=split,
                args_for_task={
                    "metric": metric,
                    "input_column": hypothesis,
                    "label_column": "label",
                    "label_mapping": {
                        "ENTAILMENT": 0,
                        "NEUTRAL": 1,
                        "CONTRADICTION": 2
                    }
                }
            )