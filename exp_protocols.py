"""Protocols P2 (GroupKFold by city), P3 (leave-one-city-out) and P4 (temporal) for LC and full specifications."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import core
from core import *
from exp_seeds import metrics  # also loads tuned params

df = load()
res = {}


def run_split(tr, te, full):
    f = Fold(tr); ytr, yte = f.label(tr), f.label(te)
    yp, P = fit_predict(stacked(42), f.features(tr, full=full), ytr, f.features(te, full=full))
    return yte, yp, P


for full in (False, True):
    tag = 'full' if full else 'lc'
    # P2
    Y, YP, PP = [], [], []
    for tri, tei in GroupKFold(5).split(df, groups=df.city_id):
        y, yp, P = run_split(df.iloc[tri], df.iloc[tei], full); Y += list(y); YP += list(yp); PP.append(P)
    res[f'P2_{tag}'] = metrics(np.array(Y), np.array(YP), np.vstack(PP)); res[f'P2_{tag}']['correct'] = int((np.array(Y) == np.array(YP)).sum())
    print('P2', tag, res[f'P2_{tag}'], flush=True)
    # P3
    per = {}
    for c in sorted(df.city_id.unique()):
        y, yp, P = run_split(df[df.city_id != c], df[df.city_id == c], full)
        per[c] = metrics(y, yp, P)
    accs = [v['acc'] for v in per.values()]
    res[f'P3_{tag}'] = dict(acc=float(np.mean(accs)), f1=float(np.mean([v['f1'] for v in per.values()])),
                            auc=float(np.mean([v['auc'] for v in per.values()])), min=float(min(accs)), max=float(max(accs)),
                            per_city={k: v['acc'] for k, v in per.items()})
    print('P3', tag, res[f'P3_{tag}'], flush=True)
    # P4
    y, yp, P = run_split(df[df.year < 2025], df[df.year == 2025], full)
    res[f'P4_{tag}'] = metrics(y, yp, P); res[f'P4_{tag}']['correct'] = int((y == yp).sum()); res[f'P4_{tag}']['n'] = len(y)
    print('P4', tag, res[f'P4_{tag}'], flush=True)
json.dump(res, open('protocols.json', 'w'), indent=1)
