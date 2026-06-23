import json
from collections import Counter

f = 'results/test_holistic_output_only_openai_gpt_oss_20b_20260622_001232.json'
d = json.load(open(f))
recs = d['results']
print('Total rows:', len(recs))

ids = [r['id'] for r in recs]
dupes = [k for k, v in Counter(ids).items() if v > 1]
print('Duplicate ids:', dupes if dupes else 'none')

for rid in ['syn_109_omission', 'syn_122_omission', 'syn_143_omission', 'syn_270_faithful']:
    matches = [r for r in recs if r['id'] == rid]
    for m in matches:
        print(f"  {rid}: pred_label={m.get('pred_label')} pred_str={m.get('pred_str')} api_error={m.get('api_error')}")