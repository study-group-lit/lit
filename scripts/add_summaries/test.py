from datasets import Dataset

test_set = Dataset.load_from_disk("/workspace/students/lit/datasets/cnn_dataset_summaries/training_chunk_12001_12064")
print(test_set[45])