"""
Semi-synthetic China Smart Infrastructure corpus
Generation script for the MEREC-WASPAS-AI study (Section 5.1 of the manuscript).

Stage 1  City-year anchor parameters for eight anchor variables (written to anchors.csv).
Stage 2  2,000 district-quarter records: truncated Gaussian marginals, Gaussian copula,
         multiplicative seasonal factors and monotone annual drift.
Stage 3  Data-quality artefacts: missingness completely at random, rate ~ U(0.01, 0.08)
         per variable; additive Gaussian sensor noise, coefficient of variation
         ~ U(0.02, 0.30) per variable.
All stages are governed by a single random seed (42).

After generation, the script computes the MEREC weights (Eqs. 2-7), the neutrosophic
memberships and composite weights (Eqs. 9-14), the WASPAS score (Eqs. 15-17, lambda = 0.5)
and the tertile optimization class on the full corpus. Inside every validation fold these
quantities are recomputed from the training partition only (Section 7.7).

Usage:  python generate_corpus.py   (writes all outputs to ./output)
"""
import json, os
import numpy as np
import pandas as pd

SEED = 42
OUT = '../data'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

CITIES = ['Shanghai', 'Beijing', 'Shenzhen', 'Guangzhou', 'Chengdu', 'Wuhan', 'Hangzhou', 'Tianjin', 'Nanjing', "Xi'an"]
DISTRICTS = dict(zip(CITIES, [15, 15, 14, 13, 12, 12, 12, 11, 11, 10]))   # 125 districts, mean 12.5
YEARS = [2022, 2023, 2024, 2025]
QUARTERS = [1, 2, 3, 4]

CRITERIA = ['energy_consumption_kWh', 'water_usage_m3', 'traffic_density', 'air_quality_index', 'noise_level_dB',
            'green_space_ratio', 'smart_meter_coverage', 'iot_sensor_density', 'renewable_energy_pct',
            'maintenance_cost_kCNY', 'response_time_min', 'infrastructure_age_yr', 'population_density_km2',
            'digital_twin_score', 'cyber_resilience', 'coverage_5G', 'waste_management_eff',
            'carbon_emission_tons', 'flood_risk_index', 'structural_health']
BENEFIT = {'green_space_ratio', 'smart_meter_coverage', 'iot_sensor_density', 'renewable_energy_pct',
           'digital_twin_score', 'cyber_resilience', 'coverage_5G', 'waste_management_eff', 'structural_health'}
ANCHOR_VARS = ['energy_consumption_kWh', 'water_usage_m3', 'air_quality_index', 'renewable_energy_pct',
               'carbon_emission_tons', 'coverage_5G', 'iot_sensor_density', 'population_density_km2']

RANGES = dict(energy_consumption_kWh=(200, 800), water_usage_m3=(100, 600), traffic_density=(0, 100),
              air_quality_index=(30, 200), noise_level_dB=(30, 100), green_space_ratio=(0, 1),
              smart_meter_coverage=(0, 1), iot_sensor_density=(10, 150), renewable_energy_pct=(0, 100),
              maintenance_cost_kCNY=(100, 2000), response_time_min=(1, 60), infrastructure_age_yr=(2, 25),
              population_density_km2=(500, 50000), digital_twin_score=(0, 10), cyber_resilience=(0, 1),
              coverage_5G=(0, 1), waste_management_eff=(0, 1), carbon_emission_tons=(80, 500),
              flood_risk_index=(0, 10), structural_health=(0, 100))
BASE = dict(energy_consumption_kWh=(450, 85), water_usage_m3=(330, 80), traffic_density=(52, 16),
            air_quality_index=(92, 24), noise_level_dB=(62, 9), green_space_ratio=(.38, .10),
            smart_meter_coverage=(.62, .13), iot_sensor_density=(62, 24), renewable_energy_pct=(36, 9),
            maintenance_cost_kCNY=(900, 260), response_time_min=(18, 8), infrastructure_age_yr=(12, 4.5),
            population_density_km2=(14000, 6500), digital_twin_score=(5.4, 1.8), cyber_resilience=(.60, .14),
            coverage_5G=(.66, .14), waste_management_eff=(.64, .12), carbon_emission_tons=(270, 60),
            flood_risk_index=(4.2, 1.6), structural_health=(72, 12))
# City offsets in standard-deviation units
TECH = {'Shanghai': .55, 'Shenzhen': .55, 'Beijing': .4, 'Guangzhou': .3, 'Hangzhou': .3, 'Nanjing': .05,
        'Wuhan': 0, 'Chengdu': -.15, 'Tianjin': -.3, "Xi'an": -.4}
ENERGY = {'Shanghai': .30, 'Shenzhen': .30, 'Beijing': .15, 'Guangzhou': .12, 'Hangzhou': 0, 'Nanjing': -.05,
          'Wuhan': -.08, 'Chengdu': -.12, 'Tianjin': -.05, "Xi'an": -.15}
GREEN = {'Shanghai': -.2, 'Shenzhen': -.1, 'Beijing': -.1, 'Guangzhou': 0, 'Hangzhou': .2, 'Nanjing': .1,
         'Wuhan': 0, 'Chengdu': .1, 'Tianjin': .35, "Xi'an": .45}
TECH_VARS = ('iot_sensor_density', 'digital_twin_score', 'cyber_resilience', 'coverage_5G', 'smart_meter_coverage')
DRIFT_VARS = ('coverage_5G', 'iot_sensor_density', 'renewable_energy_pct', 'digital_twin_score')
DRIFT_PER_YEAR = {'renewable_energy_pct': 0.40}      # SD units per year; other drifted variables 0.10
DRIFT_CITY_MULT = {'Chengdu': 1.9}
SEASONAL = {'energy_consumption_kWh': {1: 1.03, 2: 1.00, 3: 1.07, 4: 0.95},
            'air_quality_index': {1: 1.08, 2: 0.97, 3: 0.98, 4: 0.97}}
COPULA_PAIRS = [('iot_sensor_density', 'digital_twin_score', .44), ('iot_sensor_density', 'cyber_resilience', .41),
                ('carbon_emission_tons', 'renewable_energy_pct', -.55), ('digital_twin_score', 'cyber_resilience', .25),
                ('energy_consumption_kWh', 'carbon_emission_tons', .30), ('coverage_5G', 'iot_sensor_density', .22),
                ('air_quality_index', 'carbon_emission_tons', .18), ('traffic_density', 'noise_level_dB', .30)]

# Neutrosophic and WASPAS constants (Section 7.4.1, Section 7.5)
N_REFERENCE, CV_THRESHOLD, MAD_Z, ALPHA, LAMBDA = 100.0, 0.35, 3.5, 0.85, 0.50


def offset(var, city):
    if var in TECH_VARS:
        return TECH[city]
    if var == 'energy_consumption_kWh':
        return ENERGY[city]
    if var == 'green_space_ratio':
        return GREEN[city]
    return 0.0


def drift(var, city, year):
    if var not in DRIFT_VARS:
        return 0.0
    return DRIFT_PER_YEAR.get(var, 0.10) * (year - 2022) * DRIFT_CITY_MULT.get(city, 1.0)


# ------------------------------------------------------------------ Stage 1: anchors
anchor_rows = []
for c in CITIES:
    for y in YEARS:
        for v in ANCHOR_VARS:
            m, s = BASE[v]
            anchor_rows.append(dict(city_id=c, year=y, variable=v,
                                    anchor_mean=round(m + s * (offset(v, c) + drift(v, c, y)), 4),
                                    anchor_sd=s, lower_bound=RANGES[v][0], upper_bound=RANGES[v][1]))
anchors = pd.DataFrame(anchor_rows)

# ------------------------------------------------------------------ Stage 2: records
rows = [(c, d, y, q) for c in CITIES for d in range(DISTRICTS[c]) for y in YEARS for q in QUARTERS]
df = pd.DataFrame(rows, columns=['city_id', 'district', 'year', 'quarter'])
k = len(CRITERIA); ix = {c: i for i, c in enumerate(CRITERIA)}
R = np.eye(k)
for a, b, r in COPULA_PAIRS:
    R[ix[a], ix[b]] = R[ix[b], ix[a]] = r
Z = rng.multivariate_normal(np.zeros(k), R, size=len(df))
for v in CRITERIA:
    m, s = BASE[v]
    o = np.array([offset(v, c) + drift(v, c, y) for c, y in zip(df.city_id, df.year)])
    x = m + s * (Z[:, ix[v]] + o)
    if v in SEASONAL:
        x = x * df.quarter.map(SEASONAL[v]).values
    lo, hi = RANGES[v]
    df[v] = np.clip(x, lo + 1e-6 * (hi - lo) + (0.01 * (hi - lo) if lo == 0 else 0), hi)
clean = df.copy()

# ------------------------------------------------------------------ Stage 3: artefacts
miss_rate = {v: rng.uniform(0.01, 0.08) for v in CRITERIA}
noise_cv = {v: rng.uniform(0.02, 0.30) for v in CRITERIA}
for v in CRITERIA:
    lo, hi = RANGES[v]
    x = df[v].values + rng.normal(0, noise_cv[v] * df[v].mean(), len(df))
    x = np.clip(x, lo + 1e-6 * (hi - lo) + (0.01 * (hi - lo) if lo == 0 else 0), hi)
    x[rng.random(len(df)) < miss_rate[v]] = np.nan
    df[v] = x

# ------------------------------------------------------------------ MEREC, neutrosophic, WASPAS (full corpus)
X = df[CRITERIA].copy()
p_missing = X.isna().mean()
Xi = X.fillna(X.median())
A = Xi.values.astype(float)
ben = np.array([v in BENEFIT for v in CRITERIA])
Nm = np.where(ben, A.min(0) / A, A / A.max(0))                           # Eqs. 2-3
L = np.abs(np.log(Nm)); n = len(CRITERIA)
S = np.log(1 + L.mean(1))                                                # Eq. 4
E = np.array([np.abs(np.log(1 + (L.sum(1) - L[:, j]) / n) - S).sum() for j in range(n)])   # Eqs. 5-6
w = E / E.sum()                                                          # Eq. 7
delta = 1 - p_missing.values
n_sensor = np.array([df.loc[X[v].notna(), 'iot_sensor_density'].median() for v in CRITERIA])
T = 1 - np.exp(-delta * n_sensor / N_REFERENCE)                           # Eq. 11
CV = Xi.std().values / Xi.mean().values
I = np.minimum(CV / CV_THRESHOLD, 1)                                     # Eq. 12
med = Xi.median().values; mad = (Xi - med).abs().median().values
z = 0.6745 * (Xi.values - med) / np.where(mad == 0, 1, mad)
p_out = (np.abs(z) > MAD_Z).mean(0)
F = np.minimum(p_missing.values + p_out, 1)                               # Eq. 13
w_star = T * (1 - I / 2) * (1 - F / 2) * w                               # Eq. 9
w_hat = w_star / w_star.sum()                                            # Eq. 10
w_bar = ALPHA * w_hat + (1 - ALPHA) / n                                  # Eq. 14
Nw = np.where(ben, A / A.max(0), A.min(0) / A)
wsm = (Nw * w_bar).sum(1); wpm = np.prod(Nw ** w_bar, 1)                 # Eqs. 15-16
df['WASPAS_score'] = LAMBDA * wsm + (1 - LAMBDA) * wpm                   # Eq. 17
t1, t2 = np.percentile(df.WASPAS_score, [100 / 3, 200 / 3])
df['optimization_class'] = np.where(df.WASPAS_score < t1, 0, np.where(df.WASPAS_score > t2, 2, 1))

# ------------------------------------------------------------------ outputs
cols = CRITERIA + ['WASPAS_score', 'optimization_class', 'city_id', 'year', 'quarter']
corpus = df[cols].copy()
corpus.to_csv(f'{OUT}/china_smart_infrastructure_corpus.csv', index=False)
corpus.to_parquet(f'{OUT}/china_smart_infrastructure_corpus.parquet', index=False)
clean[CRITERIA + ['city_id', 'year', 'quarter']].to_csv(f'{OUT}/corpus_before_artefacts.csv', index=False)
anchors.to_csv(f'{OUT}/anchors.csv', index=False)
pd.DataFrame(R, index=CRITERIA, columns=CRITERIA).to_csv(f'{OUT}/copula_correlation_matrix.csv')
pd.DataFrame(dict(criterion=CRITERIA, direction=['benefit' if b else 'cost' for b in ben],
                  merec_weight=w, p_missing=p_missing.values, delta=delta, n_sensor_median=n_sensor,
                  T=T, CV=CV, I=I, p_outlier=p_out, F=F, w_star=w_star, w_hat=w_hat, composite_weight=w_bar)
             ).round(6).to_csv(f'{OUT}/weights_full_corpus.csv', index=False)
json.dump(dict(seed=SEED, n_records=len(df), cities=CITIES, districts_per_city=DISTRICTS,
               copula_pairs=COPULA_PAIRS, drift_sd_per_year=dict(default=0.10, **DRIFT_PER_YEAR),
               drift_city_multiplier=DRIFT_CITY_MULT, seasonal_factors=SEASONAL,
               stage3_missing_rate={k: round(v, 4) for k, v in miss_rate.items()},
               stage3_noise_cv={k: round(v, 4) for k, v in noise_cv.items()},
               neutrosophic=dict(n_reference=N_REFERENCE, cv_threshold=CV_THRESHOLD, mad_z=MAD_Z, alpha=ALPHA),
               waspas_lambda=LAMBDA, tertile_cut_points_full_corpus=[round(t1, 6), round(t2, 6)]),
          open(f'{OUT}/generation_parameters.json', 'w'), indent=1)
print(corpus.shape, corpus.optimization_class.value_counts().sort_index().tolist())
