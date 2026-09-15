"""Fair re-evaluation: is the previous winner still the best, now that the pool is 49?

Bug in the previous candidate pass: its "PREV_WINNER" row was NOT the previous winner --
the newweak family grew from 4 to 6 names when lda_fe2 / gnb_ord were added, so the
candidate silently kept knn_cos, knn_sig6 and nys_rbf. Every candidate in that pass
therefore scored <= 0.94272, and the true 0.94274 configuration was never re-measured.
This script pins the exact 36-member winner and asks a single clean question per new
member: does adding it to the winner help?

Candidates are deliberately few (one pass, ~3 min), then the top 3 get 3 outer seeds.
Writes a new submission only if it beats 0.94274.
"""
import os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm

t0 = time.time()
y = (pd.read_csv("数据/train.csv")["Will_Buy_EV"] == "Yes").astype(int).values
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
auc1 = np.array([roc_auc_score(y, O[:, j]) for j in range(M)])
print(f"M = {M}")
G = lambda a: norm.ppf(rankdata(a) / (len(a) + 1.0))
GO = np.column_stack([G(O[:, j]) for j in range(M)])
GT = np.column_stack([G(T[:, j]) for j in range(M)])
SEEDS = [2024, 42, 777]
FOLDS = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in SEEDS}
_memo = {}


def cf(idx, seed=2024, C=1.0):
    key = (seed, C, tuple(sorted(idx)))
    if key in _memo:
        return _memo[key]
    o = np.zeros(n)
    for tri, vai in FOLDS[seed]:
        lr = LogisticRegression(C=C, max_iter=500).fit(GO[np.ix_(tri, idx)], y[tri])
        o[vai] = lr.decision_function(GO[np.ix_(vai, idx)])
    _memo[key] = roc_auc_score(y, o)
    return _memo[key]


# ---- the exact winner of the previous round, expressed against this 49-member pool ----
DROP = ["catnb_binned", "lgb_xreg", "stumps_fe2",                      # junk
        "gnb_cont", "sgd_hinge", "et_fe2", "rf_fe2",                  # previous newweak
        "knn_manh50",                                                 # pruned k-NN
        "lda_fe2", "gnb_ord", "knn_cos", "knn_sig6", "nys_rbf"]       # this round's new 5
missing = [x for x in DROP if x not in names]
print("not present (skipped):", missing)
DROP = [x for x in DROP if x in names]
PREV = [j for j in range(M) if names[j] not in set(DROP)]
print(f"true previous winner: k={len(PREV)}  (re-measured, seed 2024) = {cf(PREV):.5f}")

NEW5 = [x for x in ["nys_rbf", "knn_sig6", "knn_cos", "lda_fe2", "gnb_ord"] if x in names]
cands = {"PREV (k=%d)" % len(PREV): PREV}
for x in NEW5:
    cands[f"PREV + {x}"] = PREV + [names.index(x)]
cands["PREV + nys/knn_sig6"] = PREV + [names.index(x) for x in ["nys_rbf", "knn_sig6"] if x in names]
cands["PREV + all5"] = PREV + [names.index(x) for x in NEW5]
# also: could the winner shed one of its own members to do better on the bigger pool?
for x in ["knn_top5", "knn_k1000", "knn_k50", "extra", "et_raw13", "v4_bag", "nn", "linear", "lrA"]:
    if x in names and names.index(x) in PREV:
        cands[f"PREV - {x}"] = [j for j in PREV if names[j] != x]

print(f"\n=== stage 1: {len(cands)} candidates (paired folds, seed 2024) ===")
res = {}
for tag, idx in sorted(cands.items(), key=lambda x: len(x[1])):
    a = cf(idx)
    res[tag] = (a, idx)
    print(f"  {tag:26s} k={len(idx):3d}  {a:.5f}  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== ranked ===")
for tag, (a, idx) in sorted(res.items(), key=lambda x: -x[1][0]):
    print(f"  {tag:26s} k={len(idx):3d}  {a:.5f}")

top = sorted(res.items(), key=lambda x: -x[1][0])[:3]
print("\n=== stage 2: 3-seed validation of top 3 ===")
fin = {}
for tag, (a1, idx) in top:
    per = [cf(idx, s) for s in SEEDS]
    fin[tag] = (float(np.mean(per)), per, idx)
    print(f"  {tag:26s} k={len(idx):3d}  {np.mean(per):.5f}   {np.round(per,5)}", flush=True)

win = max(fin, key=lambda k: fin[k][0])
wm, wper, widx = fin[win]
lr = LogisticRegression(C=1.0, max_iter=3000).fit(GO[:, widx], y)
ins = roc_auc_score(y, lr.decision_function(GO[:, widx]))
pred = lr.decision_function(GT[:, widx])
print(f"\n>>> winner = {win}  cross-fit {wm:.5f} | in-sample {ins:.5f} | gap {wm-ins:+.5f}")
print("  coefs:", {names[j]: round(float(c), 3)
                   for j, c in sorted(zip(widx, lr.coef_[0]), key=lambda x: -abs(x[1]))})

PREV_BEST = 0.94274
out = "submission_final_stack.csv" if wm > PREV_BEST + 1e-5 else "submission_stack_v3.csv"
sub_out = pd.read_csv("数据/test.csv")[["id"]].copy()
sub_out["Will_Buy_EV"] = pd.Series(pred).rank(pct=True).values
sub_out.to_csv(out, index=False)
print(f"\n  prev best = {PREV_BEST:.5f} | winner = {wm:.5f} -> wrote {out}  n={len(sub_out)}")
print(f"total {time.time()-t0:.0f}s")
