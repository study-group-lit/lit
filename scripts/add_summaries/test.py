from datasets import Dataset
import json

test_set = Dataset.load_from_disk("/workspace/students/lit/datasets/cnn_dataset_fixed_summaries/training_chunk_0_64")
print(json.dumps(test_set[42]))