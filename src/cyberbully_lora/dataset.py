import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .labels import CLASSIFICATION_LABELS

IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ClassificationDataset(Dataset):
    """Single-label classification over the whole tweet (bully / sentiment / target)."""

    def __init__(self, jsonl_path: Path, tokenizer, task: str, max_length: int = 128):
        assert task in CLASSIFICATION_LABELS, f"unknown task {task}"
        self.task = task
        self.label_list = CLASSIFICATION_LABELS[task]
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.records = read_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        enc = self.tokenizer(
            record["tweet"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label_to_id[record[self.task]], dtype=torch.long)
        return item


class RationaleDataset(Dataset):
    """Token-level binary tagging: is each subword part of the bullying rationale."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 128):
        self.records = read_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        tokens = record["tokens"]
        word_tags = record["rationale_tags"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(IGNORE_INDEX)
            elif word_id != prev_word_id:
                labels.append(word_tags[word_id])
            else:
                labels.append(IGNORE_INDEX)  # only the first subword of each word carries the label
            prev_word_id = word_id

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item


def build_dataset(task: str, jsonl_path: Path, tokenizer, max_length: int = 128):
    if task == "rationale":
        return RationaleDataset(jsonl_path, tokenizer, max_length=max_length)
    return ClassificationDataset(jsonl_path, tokenizer, task=task, max_length=max_length)
