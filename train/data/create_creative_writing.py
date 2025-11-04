#!/usr/bin/env python3
"""
Utility script for converting the raw creative-writing CSV (stories.csv) into
the parquet format expected by the VERL RLHF dataloader.

Each row in the source CSV is expected to contain:
    - human_story: high-quality human authored story (reference solution)
    - title:      assignment / prompt description
    - genre:      optional genre metadata

Usage:
    python create_creative_writing.py --input ../../stories.csv --output creative_writing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_INSTRUCTIONS = """Write an original, polished creative piece that fully answers the assignment.
- Target length: 400-600 words (flex where needed for quality).
- You may include minimal planning, but the final piece must be clearly presented.
- Avoid copying the exemplar text; produce a fresh story in your own voice.
- Wrap the final submission in <writing>...</writing> tags.
"""


@dataclass
class Sample:
    split: str
    index: int
    prompt_text: str
    reference_story: str
    assignment_title: str
    assignment_genre: str

    def to_record(self) -> Dict[str, Any]:
        question_id = f"{self.split}_{self.index:05d}"
        extra_info = {
            "question_id": question_id,
            "index": self.index,
            "assignment": self.assignment_title,
        }
        if self.assignment_genre:
            extra_info["genre"] = self.assignment_genre

        split_key = self.split.lower()
        if split_key.startswith("train"):
            data_source = "llm_judge_creative_train"
        else:
            data_source = "llm_judge_creative_test"

        record = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": self.prompt_text,
                }
            ],
            "ability": "creative_writing",
            "reward_model": {
                "style": "llm_judge",
                "ground_truth": self.reference_story,
                "reference_story": self.reference_story,
                "assignment": self.assignment_title,
            },
            "extra_info": extra_info,
        }
        return record


def build_prompt(title: str, genre: str | float | None, reference_story: str) -> str:
    genre_text = ""
    if isinstance(genre, str) and genre.strip():
        genre_text = f"Genre or vibe hints: {genre.strip()}\n"

    exemplar_notice = (
        "You will later be evaluated against an exemplar human response. Use it as inspiration "
        "for ambition, not as text to copy."
    )

    prompt_parts = [
        f"Assignment:\n{title.strip()}",
        genre_text.strip(),
        DEFAULT_INSTRUCTIONS.strip(),
        exemplar_notice,
    ]

    return "\n\n".join(part for part in prompt_parts if part)


def split_dataframe(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(shuffled) * val_fraction)
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    train_df = shuffled.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df


def dataframe_to_records(df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        story = str(row["human_story"]).strip()
        if not story:
            continue

        title = str(row.get("title", "")).strip() or "Write an original story that fits the creative brief."
        genre = row.get("genre", "")
        prompt_text = build_prompt(title=title, genre=genre, reference_story=story)

        sample = Sample(
            split=split,
            index=idx,
            prompt_text=prompt_text,
            reference_story=story,
            assignment_title=title,
            assignment_genre=str(genre).strip() if isinstance(genre, str) else "",
        )
        records.append(sample.to_record())
    return records


def save_records(records: List[Dict[str, Any]], output_path: Path) -> None:
    if not records:
        raise ValueError(f"No records to save for {output_path}")
    df = pd.DataFrame(records)
    try:
        df.to_parquet(output_path)
    except ImportError as err:
        fallback_path = output_path.with_suffix(".jsonl")
        df.to_json(fallback_path, orient="records", lines=True)
        raise ImportError(
            f"Writing parquet requires pyarrow or fastparquet. Saved a JSONL fallback to {fallback_path}. "
            "Install `pyarrow` and rerun the script."
        ) from err


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert creative writing CSV to VERL parquet format.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input stories.csv file.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for parquet files.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of data to use for validation.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for the train/val split.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Unable to find input CSV at {args.input}")

    df = pd.read_csv(args.input)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "human_story" not in df.columns:
        raise ValueError("Input CSV must contain a 'human_story' column.")

    args.output.mkdir(parents=True, exist_ok=True)

    train_df, val_df = split_dataframe(df, val_fraction=args.val_fraction, seed=args.seed)

    train_records = dataframe_to_records(train_df, split="train")
    val_records = dataframe_to_records(val_df, split="val")

    save_records(train_records, args.output / "train.parquet")
    save_records(val_records, args.output / "val.parquet")

    # Also keep JSON for inspectability.
    (args.output / "train.json").write_text(json.dumps(train_records[:50], indent=2))
    (args.output / "val.json").write_text(json.dumps(val_records[:50], indent=2))

    print(f"Wrote {len(train_records)} training samples to {args.output / 'train.parquet'}")
    print(f"Wrote {len(val_records)} validation samples to {args.output / 'val.parquet'}")


if __name__ == "__main__":
    main()
