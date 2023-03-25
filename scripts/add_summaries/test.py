from datasets import Dataset

test_set = Dataset.load_from_disk("/workspace/students/lit/datasets/cnn_dataset_summaries/training_chunk_0_64")
print(test_set[0])