"""Core pipeline for the MEREC-WASPAS-AI study: fold-internal weighting, labelling and models."""
import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, StackingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

CRITERIA = ['energy_consumption_kWh', 'water_usage_m3', 'traffic_density', 'air_quality_index', 'noise_level_dB',
            'green_space_ratio', 'smart_meter_coverage', 'iot_sensor_density', 'renewable_energy_pct',
            'maintenance_cost_kCNY', 'response_time_min', 'infrastructure_age_yr', 'population_density_km2',
            'digital_twin_score', 'cyber_resilience', 'coverage_5G', 'waste_management_eff',
            'carbon_emission_tons', 'flood_risk_index', 'structural_health']
BENEFIT = np.array([c in {'green_space_ratio', 'smart_meter_coverage', 'iot_sensor_density', 'renewable_energy_pct',
                          'digital_twin_score', 'cyber_resilience', 'coverage_5G', 'waste_management_eff',
                          'structural_health'} for c in CRITERIA])
N_REF, CV_THR, MAD_Z, ALPHA, LAM = 100.0, 0.35, 3.5, 0.85, 0.50
n = len(CRITERIA)


def load():
    return pd.read_csv('../data/china_smart_infrastructure_corpus.csv')


# ---------------------------------------------------------------- weighting
def merec(A):
    Nm = np.where(BENEFIT, A.min(0) / A, A / A.max(0))
    L = np.abs(np.log(Nm)); S = np.log(1 + L.mean(1))
    E = np.array([np.abs(np.log(1 + (L.sum(1) - L[:, j]) / n) - S).sum() for j in range(n)])
    return E / E.sum()


def memberships(Xraw, Ximp, iot_col, n_ref=N_REF, cv_thr=CV_THR):
    p_miss = np.isnan(Xraw).mean(0)
    delta = 1 - p_miss
    nsen = np.array([np.nanmedian(iot_col[~np.isnan(Xraw[:, j])]) for j in range(n)])
    T = 1 - np.exp(-delta * nsen / n_ref)
    CV = Ximp.std(0, ddof=1) / Ximp.mean(0)
    I = np.minimum(CV / cv_thr, 1)
    med = np.median(Ximp, 0); mad = np.median(np.abs(Ximp - med), 0)
    z = 0.6745 * (Ximp - med) / np.where(mad == 0, 1, mad)
    p_out = (np.abs(z) > MAD_Z).mean(0)
    F = np.minimum(p_miss + p_out, 1)
    return dict(p_miss=p_miss, delta=delta, nsen=nsen, T=T, CV=CV, I=I, p_out=p_out, F=F)


def neutro(w, M, alpha=ALPHA):
    ws = M['T'] * (1 - M['I'] / 2) * (1 - M['F'] / 2) * w
    wh = ws / ws.sum()
    return alpha * wh + (1 - alpha) / n, ws, wh


def entropy_w(A):
    P = A / A.sum(0); E = -(P * np.log(P)).sum(0) / np.log(len(A)); d = 1 - E; return d / d.sum()


def critic_w(A):
    Nn = np.where(BENEFIT, (A - A.min(0)) / (A.max(0) - A.min(0)), (A.max(0) - A) / (A.max(0) - A.min(0)))
    s = Nn.std(0, ddof=1); R = np.corrcoef(Nn.T); C = s * (1 - R).sum(0); return C / C.sum()


class Fold:
    """All data-derived quantities estimated on a training partition and frozen."""
    def __init__(self, df_tr, scheme='neutro', lam=LAM, n_ref=N_REF, cv_thr=CV_THR, alpha=ALPHA):
        Xr = df_tr[CRITERIA].values.astype(float)
        self.med = np.nanmedian(Xr, 0)
        Xi = np.where(np.isnan(Xr), self.med, Xr)
        self.mn, self.mx = Xi.min(0), Xi.max(0)
        self.w_merec = merec(Xi)
        self.M = memberships(Xr, Xi, Xr[:, CRITERIA.index('iot_sensor_density')], n_ref, cv_thr)
        self.w_neutro, self.w_star, self.w_hat = neutro(self.w_merec, self.M, alpha)
        self.w = {'neutro': self.w_neutro, 'classical': self.w_merec, 'equal': np.full(n, 1 / n),
                  'entropy': entropy_w(Xi), 'critic': critic_w(Xi)}[scheme]
        self.lam = lam
        q = self.score(df_tr)
        self.t1, self.t2 = np.percentile(q, [100 / 3, 200 / 3])

    def impute(self, df):
        Xr = df[CRITERIA].values.astype(float)
        return np.where(np.isnan(Xr), self.med, Xr)

    def norm(self, df):
        X = np.clip(self.impute(df), self.mn, self.mx)
        return np.where(BENEFIT, X / self.mx, self.mn / X)

    def score(self, df, w=None, lam=None):
        w = self.w if w is None else w; lam = self.lam if lam is None else lam
        Nw = self.norm(df)
        return lam * (Nw * w).sum(1) + (1 - lam) * np.prod(Nw ** w, 1)

    def label(self, df, w=None):
        q = self.score(df, w)
        return np.where(q < self.t1, 0, np.where(q > self.t2, 2, 1))

    def features(self, df, full=False, cols=None):
        Xn = self.norm(df)
        if cols is not None:
            Xn = Xn[:, cols]
        if full:
            Xn = np.column_stack([Xn, self.score(df), np.tile(self.w, (len(df), 1))])
        return Xn


# ---------------------------------------------------------------- models
PARAMS = {}   # filled from tuning results


def base(name, seed=42):
    p = PARAMS.get(name, {})
    if name == 'xgb':
        return XGBClassifier(**{**dict(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.85,
                                       colsample_bytree=0.8, reg_lambda=1.0), **p}, random_state=seed, n_jobs=1,
                             tree_method='hist', verbosity=0)
    if name == 'lgbm':
        return LGBMClassifier(**{**dict(n_estimators=200, num_leaves=31, learning_rate=0.08, min_child_samples=20,
                                        colsample_bytree=0.8), **p}, random_state=seed, n_jobs=1, verbose=-1)
    if name == 'mlp':
        return MLPClassifier(**{**dict(hidden_layer_sizes=(128, 64, 32), learning_rate_init=0.001, alpha=1e-4,
                                       batch_size=64), **p}, early_stopping=True, n_iter_no_change=10,
                             max_iter=500, random_state=seed)
    if name == 'rf':
        return RandomForestClassifier(**{**dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2), **p},
                                      random_state=seed, n_jobs=1)
    if name == 'et':
        return ExtraTreesClassifier(**{**dict(n_estimators=500, max_features='sqrt', min_samples_leaf=1), **p},
                                    random_state=seed, n_jobs=1)
    if name == 'svm':
        return SVC(**{**dict(C=10.0, gamma='scale'), **p}, kernel='rbf', probability=True, random_state=seed)
    if name == 'gb':
        return GradientBoostingClassifier(**{**dict(n_estimators=200, max_depth=3, learning_rate=0.1), **p},
                                          random_state=seed)
    if name == 'ada':
        from sklearn.tree import DecisionTreeClassifier
        p = dict(p); md = p.pop('max_depth', 2)
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=md, random_state=seed),
                                  **{**dict(n_estimators=200, learning_rate=0.5), **p}, random_state=seed)
    raise ValueError(name)


ENSEMBLE = ['xgb', 'lgbm', 'mlp', 'rf', 'et']


def stacked(seed=42, members=ENSEMBLE, meta=True):
    est = [(m, base(m, seed)) for m in members]
    if len(est) == 1:
        return est[0][1]
    if not meta:
        return VotingClassifier(est, voting='soft')
    return StackingClassifier(est, final_estimator=LogisticRegression(C=1.0, max_iter=2000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), stack_method='predict_proba',
                              n_jobs=1)


COMPARATORS = {'B1': 'svm', 'B2': 'gb', 'B3': 'rf', 'B4': 'ada', 'B5': 'xgb', 'B6': 'mlp', 'B7': 'lgbm'}


def fit_predict(model, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    model.fit(sc.transform(Xtr), ytr)
    P = model.predict_proba(sc.transform(Xte))
    return P.argmax(1), P
