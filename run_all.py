"""Regenerate the corpus, all experiments, every table value and every figure.

Run from the repository root:  python run_all.py
All scripts are executed with outputs/ as the working directory. Full execution takes
about three hours on a two-core machine (20 seeds x 8 comparators x ablation variants).
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'outputs')
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'code'))
STEPS = ['generate_corpus.py',   # Stage 1-3 corpus -> data/
         'tune.py',              # hyperparameter search on the P1 training partition
         ('exp_seeds.py', '0'), ('exp_seeds.py', '1'),   # P1/P5: Tables 3, 7, 9, 11
         'exp_protocols.py',     # P2-P4: Table 10
         'exp_misc.py',          # Tables 4-6, 8, CV, lambda objective, sensitivity, worked example
         'aggregate.py',         # results.json and CSV outputs
         'eda_final.py',         # Figures 4-6
         'res_real.py',          # Figures 7-13
         'diag.py']              # Figures 1-3
for step in STEPS:
    args = list(step) if isinstance(step, tuple) else [step]
    print('>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, 'code', args[0])] + args[1:], cwd=OUT, env=env, check=True)
print('Done. Tables: outputs/results.json and CSV files; figures: figures/')
