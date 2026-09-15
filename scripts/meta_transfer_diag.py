"""Transfer-risk diagnostic: does the non-linear meta survive the OOF -> TEST
smoothness shift?

The measured problem (round 5, stack_meta3):
  member matrix mean pairwise |corr| = 0.9383 on OOF  vs  0.9519 on TEST
because each OOF column is a SINGLE 80%-model prediction while most TEST columns are
5-fold averages (or full-data refits).  So the meta is trained on noisier inputs than
it will see.  That can help (test inputs are cleaner) or hurt (the tree's split
thresholds were tuned to the wrong noise level) - and cross-fit CANNOT see it, because
cross-fit scores the meta on OOF columns too.

This script makes the question measurable instead of rhetorical.  It builds a
"test-like" OOF matrix by shrinking every member column toward the row consensus,

    G_smooth_j = (1 - lam) * G_j + lam * rowmean(G)

and picking lam so that the smoothed matrix reproduces the TEST-level mean pairwise
|corr| (0.9519).  If the cross-fit score holds up - or improves - on the smoothed
matrix, the meta is robust to the shift in the direction that actually matters.
"""

import os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm

t0 = time.time()
SEEDS = [2024, 42]
TARGET_CORR = 0.9519          # measured TEST-level mean pairwise |corr|
BASE_CORR = 0.9383            # measured OOF-level mean pairwise |corr|

tr = pd.read_csv("数据/train.csv"); te = pd.read_csv("数据/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n = len(y)

ALIAS = {"oof10_raw13_10f": "testfull_raw13_10f", "oof10_fe2_10f": "testfull_fe2_10f",
         "oof_lgb_shallow_s7": "testfull_lgb_shallow_s7",
         "oof_lgb_shallow_ord": "testfull_lgb_shallow_ord"}
disc = {}
for f in sorted(os.listdir(".")):
    if not (f.startswith("oof_") and f.endswith(".npy")) or f.startswith("oof10_"):
        continue
    k = f[:-4]
    tf = "test_" + k[4:] + ".npy"
    if os.path.exists(tf):
        ov = np.load(f)
        if len(ov) == n:
            disc[k] = (ov, np.load(tf))
for k, tf in ALIAS.items():
    if os.path.exists(k + ".npy") and os.path.exists(tf + ".npy"):
        ov = np.load(k + ".npy")
        if len(ov) == n:
            disc[k] = (ov, np.load(tf + ".npy"))
pool = [(k[4:], v[0], v[1]) for k, v in sorted(disc.items())]
names = [p[0] for p in pool]
M = len(names)
O = np.column_stack([p[1] for p in pool])
T = np.column_stack([p[2] for p in pool])
DROP = ["catnb_binned", "lgb_xreg", "stumps_fe2", "gnb_cont", "sgd_hinge", "et_fe2",
        "rf_fe2", "knn_manh50", "lda_fe2", "gnb_ord", "knn_sig6", "nys_rbf"]
DROP = [x for x in DROP if x in names]
WIDX = [j for j in range(M) if names[j] not in set(DROP)]
assert len(WIDX) == 37, len(WIDX)
gauss = lambda a: norm.ppf(rankdata(a) / (len(a) + 1.0))
G = np.column_stack([gauss(O[:, j]) for j in WIDX])
Gt = np.column_stack([gauss(T[:, j]) for j in WIDX])


def mean_abs_corr(A):
    C = np.corrcoef(A, rowvar=False)
    iu = np.triu_indices(C.shape[0], 1)
    return float(np.abs(C[iu]).mean())


print(f"base  OOF mean|corr| = {mean_abs_corr(G):.4f}   (target for TEST = {TARGET_CORR})")
print(f"base  TEST mean|corr| = {mean_abs_corr(Gt):.4f}")
print(f"col std mean: OOF {G.std(0).mean():.4f}  TEST {Gt.std(0).mean():.4f}", flush=True)

# ---- find lam reproducing the TEST correlation level on the OOF matrix ----
mean_tr = G.mean(1, keepdims=True)
best_lam, best_gap = 0.0, 1e9
for lam in np.arange(0.0, 0.60, 0.02):
    c = mean_abs_corr((1 - lam) * G + lam * mean_tr)
    if abs(c - TARGET_CORR) < best_gap:
        best_gap, best_lam = abs(c - TARGET_CORR), float(lam)
print(f"\nlam = {best_lam:.2f} reproduces mean|corr| = "
      f"{mean_abs_corr((1-best_lam)*G + best_lam*mean_tr):.4f} on OOF", flush=True)

order = np.argsort(-np.abs(LogisticRegression(C=1.0, max_iter=3000).fit(G, y).coef_[0]))[:10]
PAIRS = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]


def build(src):
    """members block + pairwise-diff block, recomputed from whatever matrix is given"""
    d = np.column_stack([src[:, a] - src[:, b] for a, b in PAIRS])
    return np.column_stack([src, d])


LT = dict(objective="binary", metric="auc", verbose=-1, linear_tree=True, num_leaves=8,
          max_depth=3, learning_rate=0.05, min_child_samples=800, lambda_l2=10.0,
          feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, random_state=42)
FOLDS = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in SEEDS}


def crossfit(X, rounds=400, seed=2024):
    p = dict(LT)
    o = np.zeros(n)
    for tri, vai in FOLDS[seed]:
        m = lgb.train(p, lgb.Dataset(X[tri], y[tri]), rounds, callbacks=[lgb.log_evaluation(0)])
        o[vai] = m.predict(X[vai])
    return o


Gsm = (1 - best_lam) * G + best_lam * mean_tr
Xb, Xs = build(G), build(Gsm)
print(f"\ndesign cols: base {Xb.shape[1]}  smoothed {Xs.shape[1]}")

print("\n=== cross-fit AUC: base OOF vs test-like (smoothed) OOF ===", flush=True)
for tag, X in (("base", Xb), ("smoothed", Xs)):
    per = [roc_auc_score(y, crossfit(X, 400, s)) for s in SEEDS]
    print(f"  {tag:9s} {np.mean(per):.5f}  {np.round(per,5)}  ({time.time()-t0:.0f}s)", flush=True)

# ---- and the direct version: fit on full OOF, predict OOF vs TEST, compare shapes ----
# NOTE (round-7 defect, fixed here): do NOT gauss() both sides before comparing quantiles.
# gauss() forces exact standard-normal marginals, so every percentile comes out identical
# and the check is degenerate / vacuously "passing".  Compare RAW probabilities instead.
print("\n=== meta output distribution, OOF vs TEST (fit on full OOF, RAW probabilities) ===", flush=True)
Xte = build(Gt)
m = lgb.train(LT, lgb.Dataset(Xb, y), 400, callbacks=[lgb.log_evaluation(0)])
po = m.predict(Xb)
pt = m.predict(Xte)
qs = [1, 5, 25, 50, 75, 95, 99]
print(f"  {'q':>4s} {'OOF':>9s} {'TEST':>9s} {'diff':>9s}")
for q in qs:
    a, b = np.percentile(po, q), np.percentile(pt, q)
    print(f"  {q:>4d} {a:9.4f} {b:9.4f} {b-a:+9.4f}")
print(f"  mean {po.mean():9.4f} {pt.mean():9.4f} {pt.mean()-po.mean():+9.4f}")
print(f"  std  {po.std():9.4f} {pt.std():9.4f} {pt.std()-po.std():+9.4f}")
print(f"\n  OOF is in-sample (optimistic) -> only the SHAPE/quantile agreement matters.")
print(f"  caveat: TEST has no labels here, so this checks input/output SHIFT, not correctness.")
print(f"total {time.time()-t0:.0f}s")
