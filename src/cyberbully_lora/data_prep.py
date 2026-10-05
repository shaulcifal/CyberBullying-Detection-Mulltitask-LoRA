"""Parse the raw BullyExplain.xlsx into a clean, unified JSONL dataset.

Source: Maity et al., "Explain Thyself Bully: Sentiment Aided Cyberbullying
Detection with Explanation" (ICDAR 2023). https://github.com/Jhaprince/BullyExplain

The raw sheet has two data-quality issues this script fixes:
1. Three trailing columns (Unnamed: 7/8/9) are leftover Excel dropdown-validation
   lists, not per-row data - they're dropped.
2. The rationale column (`Explaine`) is a lightly-edited copy of the tweet with
   `{` prepended to rationale words, but @mentions are stripped and newlines are
   sometimes collapsed without a space, so word positions don't line up 1:1 with
   the original tweet. We recover per-word rationale tags by aligning the two
   word sequences with difflib.SequenceMatcher (after removing @mentions from
   the tweet side to match) instead of assuming positional alignment.
"""
import argparse
import difflib
import json
import random
import re
from pathlib import Path

import pandas as pd

CORE_COLUMNS = ["tweet", "Bully_Label", "Sentiment", "Target", "Explaine"]
MENTION_RE = re.compile(r"^@\w+$")


def load_raw(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df[["tweet", "Bully_Label", "Sentiment", "Emotion_label", "Sarcasm", "Target", "Explaine"]]
    df = df.dropna(subset=CORE_COLUMNS).reset_index(drop=True)
    return df


def tweet_words(tweet: str) -> list[str]:
    return str(tweet).replace("\n", " ").split()


def explain_words(explain: str) -> list[str]:
    return str(explain).replace("\n", " ").split()


def align_rationale(tweet: str, explain: str) -> tuple[list[str], list[int]]:
    """Return (tweet_tokens, rationale_tags) with tags aligned via word-level diff."""
    tw = tweet_words(tweet)
    ex_raw = explain_words(explain)
    ex_flags = [w.startswith("{") for w in ex_raw]
    ex_clean = [w.lstrip("{") for w in ex_raw]

    # indices into `tw` of words that are NOT @mentions (Explaine never contains them)
    keep_idx = [i for i, w in enumerate(tw) if not MENTION_RE.match(w)]
    tw_filtered = [tw[i].lower() for i in keep_idx]
    ex_lower = [w.lower() for w in ex_clean]

    tags = [0] * len(tw)
    matcher = difflib.SequenceMatcher(None, tw_filtered, ex_lower, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            filt_pos = block.a + offset
            ex_pos = block.b + offset
            if ex_flags[ex_pos]:
                tags[keep_idx[filt_pos]] = 1
    return tw, tags


def build_records(df: pd.DataFrame) -> list[dict]:
    records = []
    total_rationale_markers = 0
    recovered_rationale_markers = 0
    for i, row in df.iterrows():
        tokens, tags = align_rationale(row["tweet"], row["Explaine"])
        n_markers = sum(1 for w in explain_words(row["Explaine"]) if w.startswith("{"))
        total_rationale_markers += n_markers
        recovered_rationale_markers += sum(tags)
        records.append(
            {
                "id": int(i),
                # a few source cells are numeric-only (e.g. a tweet that's just "1"),
                # which openpyxl/pandas read as int/float rather than str
                "tweet": str(row["tweet"]),
                "tokens": tokens,
                "rationale_tags": tags,
                "bully": str(row["Bully_Label"]),
                "sentiment": str(row["Sentiment"]),
                "target": str(row["Target"]),
            }
        )
    coverage = recovered_rationale_markers / max(total_rationale_markers, 1)
    print(f"[data_prep] rationale alignment coverage: {recovered_rationale_markers}/{total_rationale_markers} "
          f"({coverage:.1%}) of marked rationale words were recovered onto the original tweet")
    return records


def split_records(records: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/BullyExplain.xlsx"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.raw)
    print(f"[data_prep] loaded {len(df)} rows with all core fields present")

    records = build_records(df)

    print("[data_prep] label distributions:")
    for key in ["bully", "sentiment", "target"]:
        counts = pd.Series([r[key] for r in records]).value_counts()
        print(f"  {key}: {counts.to_dict()}")

    train, val, test = split_records(records, seed=args.seed)
    write_jsonl(train, args.out_dir / "train.jsonl")
    write_jsonl(val, args.out_dir / "val.jsonl")
    write_jsonl(test, args.out_dir / "test.jsonl")
    print(f"[data_prep] wrote {len(train)} train / {len(val)} val / {len(test)} test records to {args.out_dir}")


if __name__ == "__main__":
    main()
