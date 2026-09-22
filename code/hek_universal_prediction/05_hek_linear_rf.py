#!/usr/bin/env python
"""
Step 5 (Linear + RF): HEK293 universal-motif prediction with the SAME Linear and
Random Forest protocol as the EpiMap comparison (posthoc_analysis/04_linear_rf_comparison.py),
so HEK numbers are directly comparable to the 116-line universal distribution.

Protocol (identical to posthoc 04):
  features = universal FIMO count matrix (18356 x 43)
  target   = log2(HPA TPM + 1)   [Spearman is invariant to the /log2(10) rescale EpiMap uses]
  split    = 90/10, random_state=42
  scaler   = StandardScaler fit on train only
  Linear   = LinearRegression
  RF       = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
  metric   = Spearman (train + test); + 10-fold CV for robustness
Genes modeled = those with a valid HPA target (18,221; 135 have no HEK293 HPA TPM).
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

WORK  = "/stg3/data1/sam/enhancer_prediction/multi_agent/hek_universal_prediction"
GENES = "/stg3/data1/sam/enhancer_prediction/gene_ENSG.txt"
CM    = f"{WORK}/results/HEK293_count_matrix.npy"
YLOG  = f"{WORK}/results/hek_y.npy"                # log2(HPA TPM+1), NaN if missing
REF   = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis/results/linear_rf/correlations.csv"
SEED  = 42
np.random.seed(SEED)

def sp(a, b): return float(spearmanr(np.ravel(a), np.ravel(b))[0])

# ---- data ----
X = np.load(CM).astype(np.float32)
y = np.load(YLOG)
mask = np.isfinite(y)
Xm, ym = X[mask], y[mask]
print(f"[info] count matrix {X.shape}; genes modeled (HPA present) = {mask.sum()}")

idx = np.arange(Xm.shape[0])
tr, te = train_test_split(idx, test_size=0.1, random_state=SEED)

def run(model):
    sc = StandardScaler().fit(Xm[tr])
    m  = model.fit(sc.transform(Xm[tr]), ym[tr])
    return sp(ym[tr], m.predict(sc.transform(Xm[tr]))), sp(ym[te], m.predict(sc.transform(Xm[te])))

def cv(model_factory):
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    pred = np.zeros_like(ym)
    for a, b in kf.split(idx):
        sc = StandardScaler().fit(Xm[a])
        m  = model_factory().fit(sc.transform(Xm[a]), ym[a])
        pred[b] = m.predict(sc.transform(Xm[b]))
    return sp(ym, pred)

lin_tr, lin_te = run(LinearRegression())
rf_factory = lambda: RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                           min_samples_leaf=2, random_state=SEED, n_jobs=-1)
rf_tr, rf_te = run(rf_factory())
lin_cv = cv(LinearRegression)
rf_cv  = cv(rf_factory)

print(f"[Linear] train={lin_tr:.4f} test={lin_te:.4f} cv={lin_cv:.4f}")
print(f"[RF]     train={rf_tr:.4f} test={rf_te:.4f} cv={rf_cv:.4f}")

# ---- compare to 116-line EpiMap universal distribution ----
def pct(col, v):
    if not os.path.exists(REF): return None
    r = pd.read_csv(REF)
    if col not in r: return None
    return (r[col].median(), r[col].min(), r[col].max(), float((r[col] < v).mean()*100))

lines = []
for name, col, v in [("linear", "linear_universal_test", lin_te),
                     ("rf",     "rf_universal_test",     rf_te)]:
    p = pct(col, v)
    if p:
        lines.append(f"  {name:7s} HEK test={v:.4f}  -> {p[3]:.0f}th pct of 116 "
                     f"(EpiMap median={p[0]:.4f} [{p[1]:.3f},{p[2]:.3f}])")

summary = f"""HEK293 universal-regulator: Linear + Random Forest (matches EpiMap posthoc 04)
================================================================================
Features: universal FIMO count matrix {X.shape} (p<0.01), genes modeled = {mask.sum()}
Target  : log2(HPA TPM + 1)   |   split 90/10 seed {SEED}, StandardScaler

                 train     test    10-fold CV
  Linear        {lin_tr:.4f}   {lin_te:.4f}    {lin_cv:.4f}
  RandomForest  {rf_tr:.4f}   {rf_te:.4f}    {rf_cv:.4f}

COMPARISON vs 116 EpiMap lines (universal, 90/10 test Spearman)
{chr(10).join(lines) if lines else '  (reference correlations.csv not found)'}
"""
open(f"{WORK}/results/SUMMARY_linear_rf.txt", "w").write(summary)
np.savez(f"{WORK}/results/hek_linear_rf.npz",
         linear_train=lin_tr, linear_test=lin_te, linear_cv=lin_cv,
         rf_train=rf_tr, rf_test=rf_te, rf_cv=rf_cv)
print("\n" + summary)
