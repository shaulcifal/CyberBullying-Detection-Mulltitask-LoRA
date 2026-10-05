# Cyberbullying Multi-Task LoRA

Parameter-efficient fine-tuning of a single **frozen** pretrained encoder for four
downstream cyberbullying-understanding tasks, using one lightweight **LoRA adapter
per task**:

| Task       | Type                          | Labels |
|------------|-------------------------------|--------|
| `bully`     | sequence classification       | Bully / Non_bully |
| `sentiment` | sequence classification       | Negative / Neutral / Positive |
| `target`    | sequence classification       | 10 target categories (Religion, Gender, Race, ...) |
| `rationale` | token-level binary tagging    | is each word part of the phrase that justifies the label |

## Dataset

[BullyExplain](https://github.com/Jhaprince/BullyExplain) (Maity, Jha, Jain, Saha &
Bhattacharyya, *"Explain Thyself Bully: Sentiment Aided Cyberbullying Detection with
Explanation"*, ICDAR 2023) — 6,373 code-mixed Hindi-English tweets, each annotated
with a bully label, sentiment, target category, and rationale spans (the exact
phrases the human annotator judged responsible for the classification).

The raw dataset is not redistributed in this repo (`data/raw/`, `data/processed/`
are gitignored — no explicit redistribution license from the authors). Fetch it
yourself and run the prep script:

```bash
git clone https://github.com/Jhaprince/BullyExplain /tmp/BullyExplain
cp /tmp/BullyExplain/Dataset/BullyExplain.xlsx data/raw/
python -m cyberbully_lora.data_prep
```

**Data quality note**: the rationale column in the raw sheet strips @mentions and
sometimes merges words across stripped newlines, so word positions don't line up
1:1 with the original tweet. `data_prep.py` recovers per-word rationale tags with
word-level sequence alignment (`difflib.SequenceMatcher`) instead of assuming
positional alignment, and reports alignment coverage.

## Architecture

- **Backbone**: [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased),
  frozen throughout. Chosen because it's pretrained on Indian languages and
  transliterated/code-mixed text, matching this dataset's Hinglish domain (a
  generic English encoder or LLM would be a domain mismatch here).
- **Per task**: a separate LoRA adapter (rank 16, alpha 32, on the attention
  query/value projections) plus a small task head — a linear classifier on the
  `[CLS]` token for the three classification tasks, a per-token linear layer for
  rationale tagging. Only the adapter + head are trained; the backbone never
  changes, so adapters are swappable at inference time.
- **Class-weighted loss on all four tasks.** Label distributions here are skewed
  (target ranges from 2086 "Not Applicable" down to 66 "Gender", ~32:1; only ~12%
  of tokens are rationale words). Plain cross-entropy collapses to predicting the
  majority class on every task except `bully` (confirmed during development —
  target's macro-F1 was 0.05 and sentiment's was 0.24 without weighting); inverse-
  frequency class weights fixed this (target/sentiment macro-F1 rose to
  0.13/0.59 respectively after just 3 epochs, rationale token-F1 from 0 to 0.54).

## Environment

```bash
conda create -n resume python=3.10
conda activate resume
pip install torch --index-url https://download.pytorch.org/whl/cu124  # torch>=2.6 required: transformers
                                                                        # refuses non-safetensors checkpoints
                                                                        # (like MuRIL's) on older torch (CVE-2025-32434)
pip install -r requirements.txt
```

If `~/.local/lib/pythonX.Y/site-packages` (user site-packages) exists, it can leak into
the conda env and shadow its own package versions (this caused a `libcudnn.so.9` import
error during development). If you hit unexplained import/version errors, check
`python -c "import site; print(site.ENABLE_USER_SITE)"` and set `PYTHONNOUSERSITE=1`
in the env's activation hook if needed.

## Usage

```bash
export PYTHONPATH=src

python -m cyberbully_lora.data_prep

python -m cyberbully_lora.train --task bully
python -m cyberbully_lora.train --task sentiment
python -m cyberbully_lora.train --task target
python -m cyberbully_lora.train --task rationale

python -m cyberbully_lora.evaluate --task bully
```

Each run saves its adapter + head to `outputs/<task>/` (adapter weights only,
via `peft`'s `save_pretrained` — a few MB each, not a full model copy).

## Evaluation

- Classification tasks: accuracy, macro-F1, macro-precision/recall.
- Rationale: token-level F1 plus IOU-F1 (predicted vs. gold rationale span overlap
  ≥ 0.5 counts as a match), the same style of metric used in HateXplain-family
  explainability papers.

- Trainable-parameter count is logged at the start of each training run — useful
  for the "reduced trainable parameters by >99% of backbone size" framing.
