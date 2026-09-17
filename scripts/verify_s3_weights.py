"""Verify the §3 "contributes nothing" verdicts AFTER the §4 scale fix.

The notebook's §3 table originally cited "blend weight = 0" for LR / target-encoding /
CatBoost / original-dataset members.  §4 then showed those weight readings were artefacts
of an unnormalised blend, so the claims have to be re-measured or removed.

This script measures the honest replacement: the **leave-one-out delta of a global linear
stack in gaussianised-rank space** (the corrected protocol), on the full discovered member
pool.  Members whose removal costs ~0 are the ones that genuinely add nothing.

    python -u verify_s3_weights.py          # ~9 members, ~2 min
    WIDE=1 python -u verify_s3_weights.py   # also sweep every member, ~40 min

Writes: s3_weight_check.log  (nothing else is touched)
"""
import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm

WIDE = os.environ.get("WIDE", "0") == "1"

t0 = time.time()
SEEDS = [2024, 42]
TOPIC = ["oof_lr", "oof_lrA", "oof_cte", "oof_cat_ctr4", "oof_cat_fe2",
         "oof_joint_orig", "oof_catnb_binned", "oof_nn", "oof_mono"]

tr = pd.read_csv("数据/train.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n = len(y)

# ---- discover every honest member: oof_*.npy with a matching test_*.npy ----
ALIAS = {"oof10_raw13_10f": "testfull_raw13_10f", "oof10_fe2_10f": "testfull_fe2_10f",
         "oof_lgb_shallow_s7": "testfull_lgb_shallow_s7",
         "oof_lgb_shallow_ord": "testfull_lgb_shallow_ord"}
disc = {}
for f in sorted(os.listdir(".")):
    if not (f.startswith("oof_") and f.endswith(".npy")) or f.startswith("oof10_"):
        continue
    k = f[:-4]
    if k.startswith("oof_meta"):        # derived meta learners -> circular, exclude
        continue
    if os.path.exists("test_" + k[4:] + ".npy") and len(np.load(f)) == n:
        disc[k] = np.load(f)
for k, tf in ALIAS.items():
    if os.path.exists(k + ".npy") and os.path.exists(tf + ".npy") and len(np.load(k + ".npy")) == n:
        disc[k] = np.load(k + ".npy")

names = sorted(disc)
G = np.column_stack([norm.ppf(rankdata(disc[k]) / (n + 1)) for k in names]).astype(np.float64)
print(f"pool = {len(names)} members   ({time.time()-t0:.0f}s)")

FOLDS = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(G, y)) for s in SEEDS}


def cf(cols=None, seeds=SEEDS):
    """cross-fitted LR stacking AUC over `cols` (indices); mean over seeds"""
    idx = np.arange(G.shape[1]) if cols is None else np.asarray(cols)
    A = G[:, idx]
    out = []
    for s in seeds:
        o = np.zeros(n)
        for tri, vai in FOLDS[s]:
            m = LogisticRegression(C=1.0, max_iter=2000).fit(A[tri], y[tri])
            o[vai] = m.predict_proba(A[vai])[:, 1]
        out.append(roc_auc_score(y, o))
    return float(np.mean(out)), out


base, _ = cf()
print(f"\nfull   pool ({len(names):2d} members): {base:.5f}")
print(f"\n{'member':22s} {'standalone':>10s} {'LOO delta':>11s} {'coef':>8s}  verdict")
print("-" * 78)

full_lr = LogisticRegression(C=1.0, max_iter=2000).fit(G, y)
coef = dict(zip(names, full_lr.coef_[0]))

allidx = list(range(len(names)))
rows = []
for k in names:
    if k not in TOPIC:
        continue
    j = names.index(k)
    a, _ = cf([i for i in allidx if i != j])
    d = base - a
    stand = roc_auc_score(y, disc[k])
    verdict = ("genuinely adds nothing" if d < 5e-5 else
               "small but real contribution" if d < 2e-4 else "contributes materially")
    print(f"{k:22s} {stand:10.5f} {d:+11.5f} {coef[k]:+8.3f}  {verdict}")
    rows.append((k, stand, d, coef[k]))

print("\n--- wider scan: every member whose removal is free or profitably removed ---")
if not WIDE:
    print("  (skipped; set WIDE=1 to sweep all members - ~40 min)")
else:
    free, costly = [], []
    for k in names:
        if k in TOPIC:
            continue
        j = names.index(k)
        a, _ = cf([i for i in allidx if i != j])
        d = base - a
        (free if d <= 5e-5 else costly).append((k, d))
    free.sort(key=lambda r: r[1])
    costly.sort(key=lambda r: -r[1])
    print(f"  {len(free)} of {len(names)} members are removable at < 5e-5 cost")
    print("  5 most expensive to remove (highest marginal value):")
    for k, d in costly[:5]:
        print(f"    {k:22s} {d:+.5f}")
    print("  5 cheapest to remove (lowest marginal value):")
    for k, d in free[:5]:
        print(f"    {k:22s} {d:+.5f}")

pd.DataFrame(rows, columns=["member", "standalone_auc", "loo_delta", "lr_coef"]).to_csv(
    "s3_weight_check.csv", index=False)
print(f"\nwrote s3_weight_check.csv   total {time.time()-t0:.0f}s")
