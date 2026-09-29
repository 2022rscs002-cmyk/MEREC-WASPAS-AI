"""Tables 4-6 and 8, CV stability, lambda objective, sensitivity grid, worked example, proxy correlations, EDA."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr, kruskal
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix,
                             cohen_kappa_score, log_loss)
from sklearn.linear_model import LogisticRegression
import core
from core import *
from exp_seeds import metrics

out = {}
df = load()
tr, te = train_test_split(df, test_size=0.2, stratify=df.optimization_class, random_state=42)
f = Fold(tr); ytr, yte = f.label(tr), f.label(te)
D = pickle.load(open('p1_predictions.pkl', 'rb')); PRED = D['PRED']; city = D['city']
assert (D['yte'] == yte).all()


def wilson(k, n_, z=1.959964):
    p = k / n_; d = 1 + z * z / n_; c = (p + z * z / (2 * n_)) / d
    h = z * np.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_)) / d
    return [c - h, c + h]


# ---------------- Table 4
yp0, P0 = PRED['proposed']
T4 = {}
names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
cities = np.unique(city); idx_by_city = {c: np.where(city == c)[0] for c in cities}
rng = np.random.default_rng(2024)
boot_idx = []
for _ in range(10000):
    cs = rng.choice(cities, len(cities), replace=True)
    boot_idx.append(np.concatenate([idx_by_city[c] for c in cs]))
T4['proposed'] = dict(correct=int((yp0 == yte).sum()), wilson=wilson((yp0 == yte).sum(), len(yte)))
for b in names:
    yp, P = PRED[b]
    c0, c1 = (yp0 == yte), (yp == yte)
    bb, cc = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if bb + cc > 0 else 1.0
    dF, dA = [], []
    for ix in boot_idx:
        dF.append(f1_score(yte[ix], yp0[ix], average='macro') - f1_score(yte[ix], yp[ix], average='macro'))
    for ix in boot_idx[:2000]:
        try:
            dA.append(roc_auc_score(yte[ix], P0[ix], multi_class='ovr') - roc_auc_score(yte[ix], P[ix], multi_class='ovr'))
        except ValueError:
            pass
    T4[b] = dict(correct=int(c1.sum()), wilson=wilson(c1.sum(), len(yte)), b=bb, c=cc, p=p,
                 dF=[f1_score(yte, yp0, average='macro') - f1_score(yte, yp, average='macro')] + list(np.percentile(dF, [2.5, 97.5])),
                 dA=[roc_auc_score(yte, P0, multi_class='ovr') - roc_auc_score(yte, P, multi_class='ovr')] + list(np.percentile(dA, [2.5, 97.5])))
    print(b, T4[b], flush=True)
ps = [T4[b]['p'] for b in names]; order = np.argsort(ps); run = 0
for r, i in enumerate(order):
    run = max(run, min(1, (len(ps) - r) * ps[i])); T4[names[i]]['holm'] = run
out['T4'] = T4

# ---------------- Table 5 / confusion / Table 6
cm = confusion_matrix(yte, yp0)
out['cm'] = cm.tolist()
out['T5'] = dict(prec=precision_score(yte, yp0, average=None).tolist(), rec=recall_score(yte, yp0, average=None).tolist(),
                 f1=f1_score(yte, yp0, average=None).tolist(), support=np.bincount(yte).tolist(),
                 auc=[roc_auc_score((yte == k).astype(int), P0[:, k]) for k in range(3)],
                 w_prec=precision_score(yte, yp0, average='weighted'), w_rec=recall_score(yte, yp0, average='weighted'),
                 w_f1=f1_score(yte, yp0, average='weighted'), w_auc=roc_auc_score(yte, P0, multi_class='ovr', average='weighted'))
Y1 = np.eye(3)[yte]
brier = float(((P0 - Y1) ** 2).sum(1).mean()); ll = float(log_loss(yte, P0))
conf = P0.max(1); corr = (P0.argmax(1) == yte).astype(float); bins = np.linspace(0, 1, 16)
ece = 0; mce = 0
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.sum():
        g = abs(conf[m].mean() - corr[m].mean()); ece += m.mean() * g; mce = max(mce, g)
pc = np.clip(P0, 1e-6, 1 - 1e-6); lo_ = np.log(pc / (1 - pc)).ravel(); yy = Y1.ravel()
lr = LogisticRegression(C=1e6, max_iter=1000).fit(lo_.reshape(-1, 1), yy)
out['T6'] = dict(brier=brier, logloss=ll, ece=float(ece), mce=float(mce), slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))
print('T5', out['T5'], 'T6', out['T6'], flush=True)

# ---------------- Table 8
T8 = {}
for sch in ['classical', 'entropy', 'critic', 'equal']:
    fs = Fold(tr, scheme=sch); ys_tr, ys_te = fs.label(tr), fs.label(te)
    yp, P = fit_predict(stacked(42), fs.features(tr), ys_tr, fs.features(te))
    T8[sch] = dict(agree=float((ys_te == yte).mean()), kappa=float(cohen_kappa_score(ys_te, yte)),
                   changed=int((ys_te != yte).sum()), two=int((np.abs(ys_te - yte) == 2).sum()),
                   acc_own=float((yp == ys_te).mean()))
    print(sch, T8[sch], flush=True)
out['T8'] = T8

# ---------------- CV stability (Figure 13)
cvres = {}
for key, mk in [('proposed', lambda: stacked(42)), ('B1', lambda: base('svm')), ('B2', lambda: base('gb')), ('B3', lambda: base('rf'))]:
    Xa = f.features(tr); s = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Xa, ytr):
        yp, _ = fit_predict(mk(), Xa[a], ytr[a], Xa[b]); s.append(f1_score(ytr[b], yp, average='macro'))
    cvres[key] = s; print(key, np.mean(s), np.std(s, ddof=1), flush=True)
out['cv'] = cvres

# ---------------- lambda objective (Eq. 18) on frozen training labels
lam_grid = np.round(np.arange(0, 1.0001, 0.05), 2); obj = []; mean_q = []; sd_q = []
for L in lam_grid:
    q = f.score(tr, lam=L); obj.append(float(np.mean([((q[ytr == k] - q[ytr == k].mean()) ** 2).sum() for k in range(3)]) / len(q) * 3))
    qa = f.score(df, lam=L); mean_q.append(float(qa.mean())); sd_q.append(float(qa.std()))
out['lambda'] = dict(grid=lam_grid.tolist(), obj=obj, mean=mean_q, sd=sd_q)
print('lambda obj', dict(zip(lam_grid.tolist(), np.round(obj, 6))), flush=True)

# ---------------- sensitivity grid (Section 7.4.1)
base_w = f.w_neutro; base_rho = spearmanr(f.w_neutro, f.w_merec)[0]
dw, drho, dcls = 0, 0, 0
grid = [('alpha', a) for a in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]] + [('cv_thr', c) for c in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]] + \
       [('n_ref', r) for r in [50, 75, 100, 125, 150]]
sens = []
for k, v in grid:
    kw = dict(alpha=ALPHA, cv_thr=CV_THR, n_ref=N_REF); kw[k] = v
    g = Fold(tr, **kw)
    a = float(np.abs(g.w_neutro - base_w).max()); r = float(abs(spearmanr(g.w_neutro, g.w_merec)[0] - base_rho))
    c = float((g.label(te) != yte).mean()); sens.append(dict(param=k, value=v, max_dw=a, d_rho=r, class_change=c))
    dw, drho, dcls = max(dw, a), max(drho, r), max(dcls, c)
out['sens'] = dict(max_dw=dw, max_drho=drho, max_class_change=dcls, grid=sens)
print('sens', dw, drho, dcls, flush=True)

# ---------------- worked example for air_quality_index (P1 training partition)
j = CRITERIA.index('air_quality_index'); M = f.M
Xr = tr[CRITERIA].values.astype(float); Xi = f.impute(tr)
out['example'] = dict(w=float(f.w_merec[j]), p_miss=float(M['p_miss'][j]), delta=float(M['delta'][j]), nsen=float(M['nsen'][j]),
                      T=float(M['T'][j]), sd=float(Xi[:, j].std(ddof=1)), mean=float(Xi[:, j].mean()), CV=float(M['CV'][j]),
                      I=float(M['I'][j]), p_out=float(M['p_out'][j]), F=float(M['F'][j]), w_star=float(f.w_star[j]),
                      sum_w_star=float(f.w_star.sum()), w_hat=float(f.w_hat[j]), w_bar=float(f.w_neutro[j]),
                      nsen_range=[float(M['nsen'].min()), float(M['nsen'].max())], delta_range=[float(M['delta'].min()), float(M['delta'].max())],
                      weights=dict(zip(CRITERIA, f.w_neutro.round(4).tolist())), merec=dict(zip(CRITERIA, f.w_merec.round(4).tolist())),
                      I_all=dict(zip(CRITERIA, M['I'].round(4).tolist())))
print('example', out['example'], flush=True)

# ---------------- proxy correlations of comparator composite scores with WASPAS (training partition, equal weights)
A = f.impute(tr); A = np.clip(A, f.mn, f.mx); w = np.full(n, 1 / n); q = f.score(tr)
Nv = A / np.sqrt((A ** 2).sum(0)); V = Nv * w
best = np.where(BENEFIT, V.max(0), V.min(0)); worst = np.where(BENEFIT, V.min(0), V.max(0))
dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1)); topsis = dm / (dp + dm)
fstar = np.where(BENEFIT, A.max(0), A.min(0)); fm = np.where(BENEFIT, A.min(0), A.max(0))
Sv = (w * (fstar - A) / (fstar - fm)).sum(1); Rv = (w * (fstar - A) / (fstar - fm)).max(1)
vikor = -(0.5 * (Sv - Sv.min()) / (Sv.max() - Sv.min()) + 0.5 * (Rv - Rv.min()) / (Rv.max() - Rv.min()))
Dn = A / A.sum(0) * w; Sp = Dn[:, BENEFIT].sum(1); Sm = Dn[:, ~BENEFIT].sum(1); copras = Sp + Sm.min() * Sm.sum() / (Sm * (Sm.min() / Sm).sum())
Nn = np.where(BENEFIT, (A - A.min(0)) / (A.max(0) - A.min(0)), (A.max(0) - A) / (A.max(0) - A.min(0)))
wsm_eq = (Nn * w).sum(1); crit = (Nn * critic_w(A)).sum(1)
prom = np.zeros(len(A)); elec = np.zeros(len(A))
for jj in range(n):
    d = Nn[:, jj][:, None] - Nn[:, jj][None, :]
    prom += w[jj] * (np.clip(d, 0, None).sum(1) - np.clip(-d, 0, None).sum(1))
    elec += w[jj] * ((d >= 0).sum(1) - (d <= 0).sum(1))
out['proxy'] = {k: float(spearmanr(v, q)[0]) for k, v in dict(TOPSIS=topsis, VIKOR=vikor, COPRAS=copras, AHP_WSM=wsm_eq,
                                                               CRITIC=crit, PROMETHEE=prom, ELECTRE=elec).items()}
print('proxy', out['proxy'], flush=True)
json.dump(out, open('misc.json', 'w'), indent=1, default=float)
