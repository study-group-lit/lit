import evaluate
from evaluate.evaluation_suite import SubTask

class Suite(evaluate.EvaluationSuite):
    def __init__(self, name):
        super().__init__(name)

        self.suite = [
            self.task_for("sick", "validation", "sentence_A", "sentence_B"),
            self.task_for("multi_nli", "validation_matched"),
            self.task_for("multi_nli", "validation_mismatched")
        ]
    
    def task_for(self, data: str, split: str, premise: str = "premise", hypothesis: str = "hypothesis"):
        return SubTask(
                task_type="text-classification",
                data=data,
                split=split,
                subset=split,
                args_for_task={
                    "metric": evaluate.combine(["accuracy", "matthews_correlation"]),
                    "input_column": premise,
                    "second_input_column": hypothesis,
                    "label_column": "label",
                    "label_mapping": {
                        "ENTAILMENT": 0,
                        "NEUTRAL": 1,
                        "CONTRADICTION": 2
                    }
                }
            )