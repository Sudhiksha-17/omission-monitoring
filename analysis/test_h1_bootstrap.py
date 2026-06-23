import sys, json, glob
sys.path.insert(0, 'eval')
sys.path.insert(0, 'data')
sys.path.insert(0, 'monitors')
from metrics import paired_bootstrap_comparison, dissociation_test

def load(style, level, model_tag):
    for f in sorted(glob.glob(f'results/test_{style}_{level}*{model_tag}*.json'), reverse=True):
        d = json.load(open(f))
        if 'results' in d:
            return [r for r in d['results'] if r.get('pred_label') is not None]
    return None

models = [
    ('gpt-4o-mini',         'gpt_4o_mini'),
    ('openai/gpt-oss-20b',  'openai_gpt_oss_20b'),
    ('openai/gpt-oss-120b', 'openai_gpt_oss_120b'),
]

print("=" * 72)
print("TEST BOOTSTRAP CIs — new test results")
print("=" * 72)

for name, tag in models:
    sf = load('structured', 'full', tag)
    hf = load('holistic',   'full', tag)
    so = load('structured', 'output_only', tag)
    ho = load('holistic',   'output_only', tag)

    print(f"\nMODEL: {name}")

    if sf and hf:
        for metric in ('ba', 'sensitivity', 'specificity'):
            r = paired_bootstrap_comparison(sf, hf, metric, n_boot=5000, seed=42)
            sig = '*' if 'CI excludes zero' in r['verdict'] else ' '
            print(f"  H1 {metric:12s} diff={r['observed_diff']:+.3f} "
                  f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] {sig} {r['verdict']}")
    else:
        print("  H1: MISSING DATA")

    if so and ho:
        for metric in ('ba', 'specificity'):
            r = paired_bootstrap_comparison(so, ho, metric, n_boot=5000, seed=42)
            sig = '*' if 'CI excludes zero' in r['verdict'] else ' '
            print(f"  output_only {metric:12s} diff={r['observed_diff']:+.3f} "
                  f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] {sig} {r['verdict']}")

    # holistic degradation
    if hf and ho:
        r = paired_bootstrap_comparison(hf, ho, 'ba', n_boot=5000, seed=42)
        sig = '*' if 'CI excludes zero' in r['verdict'] else ' '
        print(f"  holistic full->out ba  diff={r['observed_diff']:+.3f} "
              f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] {sig} {r['verdict']}")