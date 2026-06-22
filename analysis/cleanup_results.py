"""
analysis/cleanup_results.py — move OSS files from subdirs, remove old files,
deduplicate keeping most recent per condition.

Run from repo root:
  python analysis/cleanup_results.py
"""
import os, shutil, glob, re
from pathlib import Path
from collections import defaultdict

results = Path("results")

# 1. Move OSS files from subdirs to results/ with sanitized names
moved = 0
for f in list(results.rglob("*.json")):
    if f.parent != results:
        new_name = f.name
        new_path = results / new_name
        if new_path.exists():
            existing_ts = new_path.stem[-15:]
            incoming_ts = f.stem[-15:]
            if incoming_ts > existing_ts:
                new_path.unlink()
                shutil.copy2(f, new_path)
                print(f"replaced: {new_path.name}")
            else:
                print(f"kept existing: {new_path.name}")
        else:
            shutil.copy2(f, new_path)
            print(f"moved: {new_path.name}")
        moved += 1

# 2. Delete all subdirs under results/
for item in results.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
        print(f"removed subdir: {item.name}")

# 3. Delete old June 19 files (old dataset, old task design)
for f in list(results.glob("*20260619*.json")):
    f.unlink()
    print(f"deleted old: {f.name}")

# 4. Delete old-format files without style/level in name
stale = [
    "dev_holistic_llama3_latest_20260619_010727.json",
    "dev_structured_llama3_latest_20260619_010727.json",
    "dev_paired_comparison_llama3_latest_20260619_010727.json",
    "dev_H1_paired_llama3_latest_20260619_020243.json",
]
for name in stale:
    p = results / name
    if p.exists():
        p.unlink()
        print(f"deleted stale: {name}")

# 5. Deduplicate: per condition keep only the most recent file
def condition_key(fname):
    m = re.match(
        r'(dev|test)_(structured|holistic|baseline|H1|dissociation)_(.+?)_(\d{8}_\d{6})\.json',
        fname)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return None

by_condition = defaultdict(list)
for f in results.glob("*.json"):
    key = condition_key(f.name)
    if key:
        by_condition[key].append(f)

deleted = 0
for key, files in by_condition.items():
    if len(files) > 1:
        # sort by timestamp descending, keep first
        files.sort(
            key=lambda f: f.stem[-15:],
            reverse=True
        )
        for old_f in files[1:]:
            old_f.unlink()
            print(f"deduped (old): {old_f.name}")
            deleted += 1

remaining = list(results.glob("*.json"))
print(f"\nDone. Moved={moved}, deduped={deleted}.")
print(f"Remaining files: {len(remaining)}")
for f in sorted(remaining):
    print(f"  {f.name}")