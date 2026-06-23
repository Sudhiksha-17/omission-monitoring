"""
schema.py — record schema, validation, and strict split handling.

One JSON record per example (see Section 3 of the spec). A record is one
(source, output) pair under one condition, with an exact label.

Guardrail note (Section 6.1 / 6.2): the loader enforces that `test` is a
distinct frozen split and refuses to silently mix splits. Few-shot example
selection must draw from train/dev only; `iter_split` makes that explicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

Condition = Literal["faithful", "omission", "adversarial_omission"]
Split = Literal["train", "dev", "test"]

VALID_CONDITIONS = ("faithful", "omission", "adversarial_omission")
VALID_SPLITS = ("train", "dev", "test")

# Binary label convention. Keep this the single source of truth.
LABEL_FAITHFUL = 0   # no safety-relevant claim omitted or downplayed
LABEL_OMISSION = 1   # a safety-relevant claim was omitted or downplayed
ID2LABEL = {0: "faithful", 1: "omission"}


@dataclass
class Claim:
    claim_id: str
    text: str                 # how the claim appears in the SOURCE document
    is_safety_relevant: bool
    # how the claim is phrased in the policy output when present. Kept distinct
    # from `text` so surface overlap between source and output is paraphrastic,
    # not exact-string — otherwise a trivial string-match baseline solves v0.
    summary_text: str = ""


@dataclass
class Record:
    id: str
    domain: str                       # "synthetic" | "sec" | "other"
    source_text: str
    claims: list[Claim]
    target_claim_id: str              # the safety-relevant claim of interest
    policy_output: str
    condition: str                    # one of VALID_CONDITIONS
    label: int                        # LABEL_FAITHFUL | LABEL_OMISSION
    split: str                        # one of VALID_SPLITS
    pair_id: str = ""                 # matched faithful/omission share a pair_id
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d: dict) -> "Record":
        claims = [Claim(**c) for c in d.get("claims", [])]
        d = {**d, "claims": claims}
        return Record(**d)


def validate(rec: Record) -> None:
    """Raise ValueError on any malformed record. Called by the loader."""
    if rec.condition not in VALID_CONDITIONS:
        raise ValueError(f"{rec.id}: bad condition {rec.condition!r}")
    if rec.split not in VALID_SPLITS:
        raise ValueError(f"{rec.id}: bad split {rec.split!r}")
    if rec.label not in (LABEL_FAITHFUL, LABEL_OMISSION):
        raise ValueError(f"{rec.id}: bad label {rec.label!r}")

    # Label/condition consistency.
    if rec.condition == "faithful" and rec.label != LABEL_FAITHFUL:
        raise ValueError(f"{rec.id}: faithful condition must have label 0")
    if rec.condition in ("omission", "adversarial_omission") and rec.label != LABEL_OMISSION:
        raise ValueError(f"{rec.id}: omission condition must have label 1")

    # Target must exist and be safety-relevant.
    by_id = {c.claim_id: c for c in rec.claims}
    if rec.target_claim_id not in by_id:
        raise ValueError(f"{rec.id}: target {rec.target_claim_id} not in claims")
    if not by_id[rec.target_claim_id].is_safety_relevant:
        raise ValueError(f"{rec.id}: target claim is not safety-relevant")
    if not any(c.is_safety_relevant for c in rec.claims):
        raise ValueError(f"{rec.id}: no safety-relevant claim present")


def write_jsonl(records: list[Record], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")


def load_jsonl(path: str | Path) -> list[Record]:
    records: list[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = Record.from_json(json.loads(line))
            validate(rec)
            records.append(rec)
    return records


def iter_split(records: list[Record], split: str) -> list[Record]:
    """Return only records in `split`. The single approved way to access data,
    so that test is never touched accidentally during prompt design."""
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return [r for r in records if r.split == split]


def example_pool(records: list[Record]) -> list[Record]:
    """Records eligible to be used as few-shot examples: train + dev only.
    NEVER test (Section 6.2: no few-shot examples drawn from test)."""
    return [r for r in records if r.split in ("train", "dev")]


def split_summary(records: list[Record]) -> dict:
    from collections import Counter
    out = {}
    for sp in VALID_SPLITS:
        sub = iter_split(records, sp)
        out[sp] = {
            "n": len(sub),
            "by_condition": dict(Counter(r.condition for r in sub)),
            "by_domain": dict(Counter(r.domain for r in sub)),
            "pos_rate": round(sum(r.label for r in sub) / max(len(sub), 1), 3),
        }
    return out
