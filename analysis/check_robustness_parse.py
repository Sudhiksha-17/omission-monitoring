import json
from collections import defaultdict
from pathlib import Path

data = Path('analysis/robustness_dev.jsonl').read_text().splitlines()
counts = defaultdict(lambda: {'total':0,'api_err':0,'unp':0})
for line in data:
    if not line.strip():
        continue
    r = json.loads(line)
    k = (r['model'], r['style'], r['variant'])
    counts[k]['total'] += 1
    if r.get('api_error'):
        counts[k]['api_err'] += 1
    if r.get('pred_str') == 'UNPARSEABLE':
        counts[k]['unp'] += 1

print(f"  {'model':25s} {'style':12s} {'var':3s}  {'total':5s}  {'api_err':7s}  {'unp':3s}")
for k in sorted(counts):
    c = counts[k]
    flag = ' *' if c['api_err'] > 2 or c['unp'] > 2 else ''
    print(f"  {k[0]:25s} {k[1]:12s} {k[2]:3s}  {c['total']:5d}  {c['api_err']:7d}  {c['unp']:3d}{flag}")