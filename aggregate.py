"""Aggregate all experiment outputs into results.json and CSV outputs for the repository."""
import json, numpy as np, pandas as pd
from scipy.stats import t as tdist

seeds = json.load(open('seeds_part0.json')) + json.load(open('seeds_part1.json'))
seeds = sorted(seeds, key=lambda r: [42, 7, 13, 21, 29, 37, 51, 63, 77, 88, 101, 123, 137, 151, 173, 199, 211, 233, 257, 271].index(r['seed']))
assert len(seeds) == 20
misc = json.load(open('misc.json')); prot = json.load(open('protocols.json'))
tq = tdist.ppf(0.975, 19)


def ms(vals):
    v = np.array(vals, float); m, s = v.mean(), v.std(ddof=1)
    return dict(mean=m, sd=s, lo=m - tq * s / np.sqrt(len(v)), hi=m + tq * s / np.sqrt(len(v)))


R = {}
keys = list(seeds[0]['R'].keys())
p1 = seeds[0]; assert p1['seed'] == 42
R['P1'] = p1['R']
R['P5'] = {k: {m: ms([s['R'][k][m] for s in seeds]) for m in ['acc', 'f1', 'auc']} for k in keys}
R['P5_delta'] = {k: ms([s['R'][k]['acc'] - s['R']['proposed']['acc'] for s in seeds]) for k in keys}
R['identical_multiplicative_all'] = all(s['identical_multiplicative'] for s in seeds)
R['top_criteria'] = sorted(set(s['top_criterion'] for s in seeds))
# Table 9
stab = {}
for lvl in seeds[0]['stab']:
    n_ = [s['stab'][lvl]['neutro'] for s in seeds]; c_ = [s['stab'][lvl]['classical'] for s in seeds]
    stab[lvl] = dict(neutro=ms(n_), classical=ms(c_), diff=ms(np.array(n_) - np.array(c_)))
R['T9'] = stab
# Table 11
rv = {}
for sc in seeds[0]['rule_vs']:
    r_ = [s['rule_vs'][sc]['rule'] for s in seeds]; c_ = [s['rule_vs'][sc]['clf'] for s in seeds]
    rv[sc] = dict(rule=ms(r_), clf=ms(c_), diff=ms(np.array(c_) - np.array(r_)))
R['T11'] = rv
R['misc'] = misc; R['prot'] = prot
json.dump(R, open('results.json', 'w'), indent=1, default=float)

rows = []
for s in seeds:
    for k, v in s['R'].items():
        rows.append(dict(seed=s['seed'], configuration=k, **v))
pd.DataFrame(rows).to_csv('seed_results.csv', index=False)
pd.DataFrame([dict(city=c, accuracy_lc=a) for c, a in prot['P3_lc']['per_city'].items()]).to_csv('loco_results.csv', index=False)
lam = misc['lambda']; pd.DataFrame(dict(lam=lam['grid'], objective=lam['obj'], mean_score=lam['mean'])).to_csv('lambda_objective.csv', index=False)


def show(label, d):
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})


print('P1:'); [show(k, v) for k, v in R['P1'].items()]
print('P5:'); [print(k, {m: f"{v['mean']:.4f}±{v['sd']:.4f}" for m, v in d.items()}) for k, d in R['P5'].items()]
print('P5 delta:'); [print(k, f"{v['mean']*100:+.2f} [{v['lo']*100:+.2f},{v['hi']*100:+.2f}]") for k, v in R['P5_delta'].items()]
print('identical multiplicative all seeds', R['identical_multiplicative_all'], R['top_criteria'])
print('T9:'); [print(k, f"C {v['classical']['mean']:.4f}±{v['classical']['sd']:.4f} N {v['neutro']['mean']:.4f}±{v['neutro']['sd']:.4f} D {v['diff']['mean']:+.4f} [{v['diff']['lo']:+.4f},{v['diff']['hi']:+.4f}]") for k, v in stab.items()]
print('T11:'); [print(k, f"rule {v['rule']['mean']:.4f}±{v['rule']['sd']:.4f} clf {v['clf']['mean']:.4f}±{v['clf']['sd']:.4f} D {v['diff']['mean']:+.4f} [{v['diff']['lo']:+.4f},{v['diff']['hi']:+.4f}]") for k, v in rv.items()]
print('PROT', json.dumps({k: {m: (round(x, 4) if isinstance(x, float) else x) for m, x in v.items() if m != 'per_city'} for k, v in prot.items()}, indent=0))
print('per city', prot['P3_lc']['per_city'])
