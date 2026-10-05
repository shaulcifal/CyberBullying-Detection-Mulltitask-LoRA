from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel

DEFAULT_BASE_MODEL = "google/muril-base-cased"


def build_lora_backbone(
    base_model_name: str = DEFAULT_BASE_MODEL,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: tuple[str, ...] = ("query", "value"),
):
    """Load the frozen pretrained encoder and attach a fresh LoRA adapter to it.

    peft's get_peft_model freezes every base-model parameter and leaves only
    the newly-injected LoRA A/B matrices trainable.
    """
    base_model = AutoModel.from_pretrained(base_model_name)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(base_model, lora_config)


class LoraSequenceClassifier(nn.Module):
    """Frozen backbone + LoRA adapter + a linear head on the [CLS] representation.

    Used for the three whole-tweet tasks: bully detection, sentiment, target.
    """

    def __init__(
        self,
        num_labels: int,
        base_model_name: str = DEFAULT_BASE_MODEL,
        backbone=None,
        class_weights: torch.Tensor | None = None,
        **lora_kwargs,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_lora_backbone(base_model_name, **lora_kwargs)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        # several tasks here are heavily imbalanced (target has up to 32:1 class
        # ratio); without weighting the model collapses to always predicting the
        # majority class
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_repr))

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.dtype) if self.class_weights is not None else None
            loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained_adapter(cls, num_labels: int, run_dir: Path, base_model_name: str = DEFAULT_BASE_MODEL):
        base_model = AutoModel.from_pretrained(base_model_name)
        backbone = PeftModel.from_pretrained(base_model, Path(run_dir) / "lora_adapter")
        model = cls(num_labels=num_labels, backbone=backbone)
        model.classifier.load_state_dict(torch.load(Path(run_dir) / "head.pt", map_location="cpu"))
        return model


class LoraTokenTagger(nn.Module):
    """Frozen backbone + LoRA adapter + a per-token binary head for rationale extraction."""

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        num_labels: int = 2,
        backbone=None,
        class_weights: torch.Tensor | None = None,
        **lora_kwargs,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_lora_backbone(base_model_name, **lora_kwargs)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        # only ~12% of tokens are positive rationale words; without weighting the
        # model collapses to always predicting the majority (non-rationale) class
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        token_repr = outputs.last_hidden_state
        logits = self.classifier(self.dropout(token_repr))

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.dtype) if self.class_weights is not None else None
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), weight=weight, ignore_index=-100
            )
        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained_adapter(cls, run_dir: Path, base_model_name: str = DEFAULT_BASE_MODEL, num_labels: int = 2):
        base_model = AutoModel.from_pretrained(base_model_name)
        backbone = PeftModel.from_pretrained(base_model, Path(run_dir) / "lora_adapter")
        model = cls(num_labels=num_labels, backbone=backbone)
        model.classifier.load_state_dict(torch.load(Path(run_dir) / "head.pt", map_location="cpu"))
        return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
