"""Protocol P1/P5: comparative, ablation, full-feature, rule-vs-classifier and label-stability runs per seed."""
import json, sys, pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import core
from core import *

core.PARAMS.update({k: (dict(v, hidden_layer_sizes=tuple(v['hidden_layer_sizes'])) if k == 'mlp' else v)
                    for k, v in json.load(open('hpo_selected.json')).items()})
for k, v in core.PARAMS.items():
    for kk, vv in list(v.items()):
        if vv == 'None': v[kk] = None
SEEDS = [42, 7, 13, 21, 29, 37, 51, 63, 77, 88, 101, 123, 137, 151, 173, 199, 211, 233, 257, 271]
GEN = json.load(open('../data/generation_parameters.json'))
MR = np.array([GEN['stage3_missing_rate'][c] for c in CRITERIA]); MR = MR / MR.mean()
NC = np.array([GEN['stage3_noise_cv'][c] for c in CRITERIA]); NC = NC / NC.mean()
IOT3 = ['iot_sensor_density', 'digital_twin_score', 'cyber_resilience']
FIVE = IOT3 + ['coverage_5G', 'smart_meter_coverage']


def metrics(y, yp, P):
    return dict(acc=accuracy_score(y, yp), prec=precision_score(y, yp, average='macro'),
                rec=recall_score(y, yp, average='macro'), f1=f1_score(y, yp, average='macro'),
                auc=roc_auc_score(y, P, multi_class='ovr', average='macro'))


def degrade(te, f, kind, level, rng):
    d = te.copy()
    X = d[CRITERIA].values.astype(float)
    sd = np.nanstd(f.impute(te), 0)
    trmean = f.med
    for j, c in enumerate(CRITERIA):
        if kind == 'miss':
            r = min(level * MR[j], 0.95); X[rng.random(len(X)) < r, j] = np.nan
        elif kind == 'noise':
            cv = level * NC[j]; X[:, j] = X[:, j] + rng.normal(0, cv * abs(trmean[j]), len(X))
        elif kind == 'out':
            r = min(level * NC[j], 0.95); m = rng.random(len(X)) < r
            X[m, j] = X[m, j] + rng.choice([-1, 1], m.sum()) * 4 * sd[j]
    lo = np.array([1e-9] * len(CRITERIA))
    X = np.where(np.isnan(X), np.nan, np.maximum(X, 1e-6))
    d[CRITERIA] = X
    return d


def run(seed):
    df = load()
    tr, te = train_test_split(df, test_size=0.2, stratify=df.optimization_class, random_state=seed)
    f = Fold(tr); ytr = f.label(tr); yte = f.label(te)
    Xtr, Xte = f.features(tr), f.features(te)
    out = dict(seed=seed, supports=np.bincount(yte, minlength=3).tolist(), y=yte.tolist(), city=te.city_id.tolist())
    R = {}; PRED = {}

    def go(key, model, A=Xtr, B=Xte, yA=ytr, yB=yte):
        yp, P = fit_predict(model, A, yA, B)
        R[key] = metrics(yB, yp, P); PRED[key] = (yp, P)
        return model

    prop = go('proposed', stacked(seed))
    for b, m in COMPARATORS.items():
        go(b, base(m, seed))
    for m in ENSEMBLE:
        go('wo_' + m, stacked(seed, [x for x in ENSEMBLE if x != m]))
    go('softvote', stacked(seed, meta=False))
    go('xgb_lgbm', stacked(seed, ['xgb', 'lgbm']))
    # multiplicative weighting variant (labels fixed)
    go('multiplicative', stacked(seed), Xtr * f.w, Xte * f.w)
    # full-feature specification
    go('full', stacked(seed), f.features(tr, full=True), f.features(te, full=True))
    out['identical_multiplicative'] = bool((PRED['multiplicative'][0] == PRED['proposed'][0]).all())

    # ---- label stability (Table 9) and rule vs classifier on degraded inputs (Table 11)
    fc = Fold(tr, scheme='classical')
    lab_n, lab_c = f.label(te), fc.label(te)
    rng = np.random.default_rng(seed + 1000)
    stab = {}
    levels = {'miss': [0.05, 0.10, 0.20, 0.30], 'noise': [0.05, 0.10, 0.20, 0.30], 'out': [0.02, 0.05, 0.10, 0.20]}
    rule_vs = {}
    sc_prop = None
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    for kind, lv in levels.items():
        for L in lv:
            dte = degrade(te, f, kind, L, rng)
            stab[f'{kind}_{L}'] = dict(neutro=float((f.label(dte) == lab_n).mean()),
                                       classical=float((fc.label(dte) == lab_c).mean()))
            if (kind, L) in [('noise', 0.10), ('noise', 0.30), ('miss', 0.10), ('miss', 0.30), ('out', 0.10)]:
                yp = prop.predict(sc.transform(f.features(dte)))
                rule_vs[f'{kind}_{L}'] = dict(rule=float((f.label(dte) == yte).mean()), clf=float((yp == yte).mean()))
    rule_vs['complete'] = dict(rule=float((f.label(te) == yte).mean()), clf=R['proposed']['acc'])
    # unavailable criteria
    top = CRITERIA[int(np.argmax(f.w))]
    out['top_criterion'] = top
    for key, drop in [('drop_top', [top]), ('drop_iot3', IOT3), ('drop_five', FIVE)]:
        keep = [j for j, c in enumerate(CRITERIA) if c not in drop]
        dte = te.copy(); dte[drop] = np.nan
        rule = float((f.label(dte) == yte).mean())
        yp, P = fit_predict(stacked(seed), Xtr[:, keep], ytr, Xte[:, keep])
        rule_vs[key] = dict(rule=rule, clf=float((yp == yte).mean()))
    out.update(R=R, stab=stab, rule_vs=rule_vs)
    if seed == 42:
        pickle.dump(dict(PRED=PRED, yte=yte, te_index=te.index.values, city=te.city_id.values), open('p1_predictions.pkl', 'wb'))
    return out


if __name__ == '__main__':
    part = int(sys.argv[1]); seeds = SEEDS[part::2]
    res = []
    for s in seeds:
        r = run(s); res.append(r)
        print(s, round(r['R']['proposed']['acc'], 4), flush=True)
        json.dump(res, open(f'seeds_part{part}.json', 'w'))
