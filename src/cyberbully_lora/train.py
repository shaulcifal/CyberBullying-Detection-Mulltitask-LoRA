import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

from .dataset import build_dataset
from .labels import CLASSIFICATION_LABELS
from .metrics import classification_metrics, rationale_metrics
from .model import DEFAULT_BASE_MODEL, LoraSequenceClassifier, LoraTokenTagger, count_trainable_params

TASKS = ["bully", "sentiment", "target", "rationale"]


def compute_class_weights(task: str, train_ds) -> torch.Tensor:
    """Inverse-frequency class weights, so the loss doesn't let the model collapse
    to always predicting the majority class. All four tasks here are imbalanced
    to varying degrees (target has up to ~32:1 class ratio).
    """
    if task == "rationale":
        num_labels = 2
        label_seqs = (record["rationale_tags"] for record in train_ds.records)
        counts = torch.zeros(num_labels, dtype=torch.long)
        for tags in label_seqs:
            counts += torch.bincount(torch.tensor(tags), minlength=num_labels)
    else:
        label_list = CLASSIFICATION_LABELS[task]
        label_to_id = {label: i for i, label in enumerate(label_list)}
        ids = torch.tensor([label_to_id[record[task]] for record in train_ds.records])
        counts = torch.bincount(ids, minlength=len(label_list))
    return counts.sum() / (len(counts) * counts.clamp(min=1))


def build_model(task: str, base_model_name: str, lora_kwargs: dict, class_weights: torch.Tensor | None = None):
    if task == "rationale":
        return LoraTokenTagger(base_model_name=base_model_name, num_labels=2, class_weights=class_weights, **lora_kwargs)
    num_labels = len(CLASSIFICATION_LABELS[task])
    return LoraSequenceClassifier(
        num_labels=num_labels, base_model_name=base_model_name, class_weights=class_weights, **lora_kwargs
    )


def build_compute_metrics(task: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if task == "rationale":
            return rationale_metrics(preds, labels)
        return classification_metrics(preds, labels)

    return compute_metrics


def save_adapter_and_head(model, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output_dir / "lora_adapter")
    torch.save(model.classifier.state_dict(), output_dir / "head.pt")


class SaveBestAdapterCallback(TrainerCallback):
    """Saves only the LoRA adapter + head (not the frozen backbone) whenever the
    tracked validation metric improves. This sidesteps Trainer's default full-model
    checkpointing, which is both unnecessary here (the backbone never changes) and
    broken for peft models: their frozen base-layer weight tensors aren't contiguous,
    which safetensors refuses to serialize.
    """

    def __init__(self, output_dir: Path, metric_key: str):
        self.output_dir = output_dir
        self.metric_key = f"eval_{metric_key}"
        self.best_score = float("-inf")
        self.best_metrics = None

    def on_evaluate(self, args, state, control, metrics, model, **kwargs):
        score = metrics.get(self.metric_key)
        if score is not None and score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics
            save_adapter_and_head(model, self.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir or Path("outputs") / args.task

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_ds = build_dataset(args.task, args.data_dir / "train.jsonl", tokenizer, max_length=args.max_length)
    val_ds = build_dataset(args.task, args.data_dir / "val.jsonl", tokenizer, max_length=args.max_length)

    lora_kwargs = dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    class_weights = compute_class_weights(args.task, train_ds)
    print(f"[train:{args.task}] class weights: {class_weights.tolist()}")
    model = build_model(args.task, args.base_model, lora_kwargs, class_weights=class_weights)

    trainable, total = count_trainable_params(model)
    print(f"[train:{args.task}] trainable params: {trainable:,} / {total:,} ({trainable / total:.3%})")

    metric_for_best = "macro_f1" if args.task != "rationale" else "token_f1"

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",  # we save adapter-only checkpoints ourselves; see SaveBestAdapterCallback
        logging_steps=20,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    best_adapter_cb = SaveBestAdapterCallback(output_dir, metric_key=metric_for_best)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=build_compute_metrics(args.task),
        callbacks=[best_adapter_cb],
    )

    trainer.train()
    print(f"[train:{args.task}] best val metrics (epoch with highest {metric_for_best}): {best_adapter_cb.best_metrics}")

    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump(best_adapter_cb.best_metrics, f, indent=2)
    print(f"[train:{args.task}] saved best LoRA adapter + head to {output_dir}")


if __name__ == "__main__":
    main()
