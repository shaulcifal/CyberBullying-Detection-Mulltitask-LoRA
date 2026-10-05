import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import build_dataset
from .labels import CLASSIFICATION_LABELS
from .metrics import classification_metrics, rationale_metrics
from .model import DEFAULT_BASE_MODEL, LoraSequenceClassifier, LoraTokenTagger
from .train import TASKS


def load_model(task: str, run_dir: Path, base_model_name: str):
    if task == "rationale":
        model = LoraTokenTagger.from_pretrained_adapter(run_dir, base_model_name=base_model_name, num_labels=2)
    else:
        num_labels = len(CLASSIFICATION_LABELS[task])
        model = LoraSequenceClassifier.from_pretrained_adapter(num_labels, run_dir, base_model_name=base_model_name)
    model.eval()
    return model


@torch.no_grad()
def run_eval(task: str, run_dir: Path, data_dir: Path, base_model_name: str, batch_size: int = 32, max_length: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_ds = build_dataset(task, data_dir / "test.jsonl", tokenizer, max_length=max_length)
    loader = DataLoader(test_ds, batch_size=batch_size)

    model = load_model(task, run_dir, base_model_name).to(device)

    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out["logits"].cpu().numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(logits, axis=-1)

    if task == "rationale":
        metrics = rationale_metrics(preds, labels)
    else:
        metrics = classification_metrics(preds, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    run_dir = args.run_dir or Path("outputs") / args.task
    metrics = run_eval(args.task, run_dir, args.data_dir, args.base_model, args.batch_size, args.max_length)
    print(f"[evaluate:{args.task}] test metrics: {metrics}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
