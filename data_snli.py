# 🔹 SNLI (Stanford Natural Language Inference)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
from functools import partial

raw_dataset = load_dataset("stanfordnlp/snli")
# print(raw_dataset["train"][0])

# Remove examples with label -1 (invalid)
def filter_labels(example):
    return example["label"] != -1

filtered_dataset = raw_dataset.filter(filter_labels)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example, tokenizer, sentence1_key, sentence2_key=None):
    if sentence2_key is None:
        return tokenizer(example[sentence1_key], truncation=True)
    else:
        return tokenizer(example[sentence1_key], example[sentence2_key], truncation=True)

tokenized_datasets = filtered_dataset .map(partial(tokenize_function, tokenizer=tokenizer, sentence1_key="premise", sentence2_key="hypothesis"), batched=True)
# print(tokenized_datasets)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# unique_labels = set(tokenized_datasets["train"]["labels"])
# print(f"Unique labels: {unique_labels}")
# print(f"Number of unique labels: {len(unique_labels)}")

# for split in tokenized_datasets:
#     unique = set(tokenized_datasets[split]["labels"])
#     print(f"{split} split - Unique labels: {unique} (Total: {len(unique)})")


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)

tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator,
)

# print(f'TF Train Dataset: {tf_train_dataset}')

