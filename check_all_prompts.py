import sys
from pathlib import Path
sys.path.insert(0, str(Path("data")))
sys.path.insert(0, str(Path("monitors")))

from schema import load_jsonl
import source_access as sa

recs = list(load_jsonl("data/dataset.jsonl"))
dev = [r for r in recs if r.split == "dev"]
examples = dev
record = dev[0]

styles = ("holistic", "structured")
levels = ("full", "claims_only_untagged", "output_only")

for style in styles:
    for level in levels:
        print("\n" + "#"*78)
        print(f"# STYLE = {style}   LEVEL = {level}")
        print("#"*78)
        # system prompt only, so we can read the instruction without the long source
        sysp = sa.get_system_prompt(style, level, "gpt-4o-mini")
        print("\n--- SYSTEM PROMPT ---")
        print(sysp)
