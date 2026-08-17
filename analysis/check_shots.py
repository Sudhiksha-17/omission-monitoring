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

for style in ("holistic", "structured"):
    prompt = sa.build_prompt(record, examples, style, "output_only", model="gpt-4o-mini", n_shots=3)
    print("\n" + "="*70)
    print("STYLE =", style, "LEVEL = output_only")
    print("="*70)
    print(prompt)