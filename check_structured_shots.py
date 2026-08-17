import sys
from pathlib import Path
sys.path.insert(0, str(Path("data")))
sys.path.insert(0, str(Path("monitors")))
from schema import load_jsonl
import source_access as sa

recs = list(load_jsonl("data/dataset.jsonl"))
dev = [r for r in recs if r.split == "dev"]
record = dev[0]

for level in ("full", "claims_only_untagged"):
    print("\n" + "#"*78)
    print(f"# STRUCTURED few-shot, LEVEL = {level}")
    print("#"*78)
    block = sa.build_few_shot_block(dev, 3, "structured", level, eval_record=record, seed=42)
    print(block)
