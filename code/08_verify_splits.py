"""
08_verify_splits.py
Audits the FROZEN train/val/test assignment in data/splits.csv.
The split is released as a fixed artefact and is NOT regenerated; this script
checks the properties the manuscript states, so a reader can verify them.

Run from the repository root:  python code/08_verify_splits.py
"""
import pandas as pd, sys

L = ['ig_outgroup', 'dehumanization', 'grievance', 'glorification', 'mobilization', 'threat']
EXPECTED = {'train': 1099, 'val': 194, 'test': 227}
DOUBLE = {'both-agree', 'double+adjudicated'}   # doubly annotated provenance

lab = pd.read_csv('data/roman_urdu_indicators_v1.csv')
sp = pd.read_csv('data/splits.csv')
d = lab.merge(sp, on='tweet_id', how='inner')
ok = True

def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))

check('every labelled post has exactly one fold', len(d) == len(lab) == sp.tweet_id.nunique(), f'{len(d)} rows')
sizes = d.fold.value_counts().to_dict()
check('fold sizes 1099/194/227', all(sizes.get(k) == v for k, v in EXPECTED.items()), str(sizes))
check('no tweet_id in more than one fold', sp.tweet_id.is_unique)
test_src = set(d[d.fold == 'test'].source)
check('test fold is entirely doubly annotated', test_src <= DOUBLE, str(sorted(test_src)))
print('\nPositive counts and prevalence (%) per fold:')
print(d.groupby('fold')[L].sum().to_string())
print((d.groupby('fold')[L].mean() * 100).round(1).to_string())
print('\nFold x provenance:')
print(pd.crosstab(d.fold, d.source).to_string())
print('\nFold x batch (split column):')
print(pd.crosstab(d.fold, d.split).to_string())
sys.exit(0 if ok else 1)
