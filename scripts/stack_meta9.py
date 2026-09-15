"""Round 9: bag over member SUBSETS instead of betting on one curated set.

Round 8 result (champion basis, paired folds):
  CUR37 0.94321 / 0.94320      <- the 37 members curated under a LINEAR combiner
  ALL49 0.94322 / 0.94321      <- everything, including the 12 the linear combiner rejected
  KNN41 0.94320 / 0.94319
  TOP30 0.94256                <- the 30 strongest members: -0.00065, a big loss

Two readings:
 1. ALL49 >= CUR37 on both seeds -> under a NON-LINEAR meta the rejected members are no
    longer harmful.  The curation was fitted to the wrong combiner, so it is a source of
    selection overfitting rather than a genuine filter.
 2. TOP30 -0.00065 -> member DIVERSEITY dominates member STRENGTH.  Pruning to the
    individually-best members is the single worst thing one can do here.

Since the three sets are statistically tied, the honest way to convert them into a gain is
not to pick one but to AVERAGE over them (random-subspace style bagging).  Each set is a
different random view of the same member pool; their errors decorrelate slightly, so the
rank-average should be >= the best single set and is far less sensitive to having picked
the "wrong" subset for the private LB.

Ships only if the bag beats 0.94320 by more than 2e-5.
"""
import os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import rankdata, norm

t0 = time.time()
SEEDS = [2024, 42]
CHAMPION = 0.94320
EPS = 2e-5

tr = pd.read_csv("数据/train.csv"); te = pd.read_csv("数据/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n, nt = len(y), len(te)

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
pool = [(k[4:], v[0], v[1]) for k, v in sorted(disc.items()) if not k.startswith("oof_meta")]
names = [p[0] for p in pool]
M = len(names)
O = np.column_stack([p[1] for p in pool])
T = np.column_stack([p[2] for p in pool])
DROP = ["catnb_binned", "lgb_xreg", "stumps_fe2", "gnb_cont", "sgd_hinge", "et_fe2",
        "rf_fe2", "knn_manh50", "lda_fe2", "gnb_ord", "knn_sig6", "nys_rbf"]
DROP = [x for x in DROP if x in names]
gauss = lambda a: norm.ppf(rankdata(a) / (len(a) + 1.0))
GALL = np.column_stack([gauss(O[:, j]) for j in range(M)])
GTALL = np.column_stack([gauss(T[:, j]) for j in range(M)])
KNNEXTRA = [x for x in ["knn_k50", "knn_k1000", "knn_manh50", "knn_sig6"] if x in names]
CUR = [k for k in names if k not in set(DROP)]
SETS = {"CUR37": CUR, "ALL49": list(names), "KNN41": CUR + KNNEXTRA}
print(f"pool M={M} | sets: " + ", ".join(f"{k}={len(v)}" for k, v in SETS.items()), flush=True)

cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level"]
raw = cat_cols + ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
                  "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
                  "Environmental_Concern_Level"]
anx = {"Low": 0, "Medium": 1, "High": 2}
CITY = {"Rural": 0, "Suburban": 1, "Urban": 2}
YESNO = {"No": 0, "Yes": 1}
ORD = {}
for c in cat_cols:
    ORD[c] = anx if c == "Range_Anxiety_Level" else (
        CITY if c == "City_Type" else (YESNO if c in ("Home_Charging_Possible", "Subsidy_Available")
                                       else {v: i for i, v in enumerate(sorted(pd.concat([tr[c], te[c]]).astype(str).unique()))}))


def orddf(df):
    d = df[raw].copy()
    for c in cat_cols:
        d[c] = d[c].astype(str).map(ORD[c]).astype(np.float64)
    return d.values.astype(np.float64)


sc = StandardScaler().fit(orddf(tr))
THREE = [raw.index(x) for x in ["Environmental_Concern_Level", "Subsidy_Available", "Annual_Income_USD"]]
S3_tr, S3_te = sc.transform(orddf(tr))[:, THREE], sc.transform(orddf(te))[:, THREE]

LT = dict(objective="binary", metric="auc", verbose=-1, linear_tree=True, num_leaves=8,
          max_depth=3, learning_rate=0.05, min_child_samples=800, lambda_l2=10.0,
          feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, random_state=42)
FOLDS = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in SEEDS}


def design(sel, G, Gt):
    """the round-5 champion basis, restricted to member set `sel` -> (A1,B1),(A2,B2),(A3,B3builder)"""
    order = np.argsort(-np.abs(LogisticRegression(C=1.0, max_iter=3000).fit(G, y).coef_[0]))[:10]
    PAIRS = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    A1 = np.column_stack([G, np.column_stack([G[:, a] - G[:, b] for a, b in PAIRS])])
    B1 = np.column_stack([Gt, np.column_stack([Gt[:, a] - Gt[:, b] for a, b in PAIRS])])
    A2, B2 = np.column_stack([G, S3_tr]), np.column_stack([Gt, S3_te])
    return (A1, B1), (A2, B2)


def oof_for(A1, A2, G, seed):
    o1 = np.zeros(n); o3 = np.zeros(n)
    for tri, vai in FOLDS[seed]:
        m = lgb.train(LT, lgb.Dataset(A1[tri], y[tri]), 400, callbacks=[lgb.log_evaluation(0)])
        o1[vai] = m.predict(A1[vai])
        # PCA must be fitted inside the fold
        pc = PCA(n_components=None, random_state=0).fit(G[tri])
        A3 = np.column_stack([G, pc.transform(G)])
        m = lgb.train(LT, lgb.Dataset(A3[tri], y[tri]), 400, callbacks=[lgb.log_evaluation(0)])
        o3[vai] = m.predict(A3[vai])
    per_seed_a1 = o1
    o2 = np.zeros(n)
    for tri, vai in FOLDS[seed]:
        m = lgb.train(LT, lgb.Dataset(A2[tri], y[tri]), 400, callbacks=[lgb.log_evaluation(0)])
        o2[vai] = m.predict(A2[vai])
    return o1, o2, o3


print("\n=== recomputing per-set OOF (needed to evaluate the bag) ===", flush=True)
Gof, Tmats, RES = {}, {}, {}
for tag, sel in SETS.items():
    idx = [names.index(k) for k in sel]
    G, Gt = GALL[:, idx], GTALL[:, idx]
    (A1, B1), (A2, B2) = design(sel, G, Gt)
    per = []
    for s in SEEDS:
        o1, o2, o3 = oof_for(A1, A2, G, s)
        per.append(np.mean([gauss(o1), gauss(o2), gauss(o3)], axis=0))
        print(f"  {tag:6s} seed{s} {roc_auc_score(y, per[-1]):.5f}  ({time.time()-t0:.0f}s)", flush=True)
    RES[tag] = per
    Gof[tag], Tmats[tag] = (A1, A2), (B1, B2)

print("\n=== individual sets vs subset-bagging ===")
best = (None, -1, None)
for tag, per in RES.items():
    # NOTE: RES[tag] holds OOF ARRAYS, not AUCs. Never pass them to a scalar formatter --
    # that was the round-9 display bug (float(np.mean(array_list)) -> 0.0 and a printed matrix).
    a = [roc_auc_score(y, p) for p in per]
    m = float(np.mean(a))
    print(f"  {tag:8s} {m:.5f} {np.round(a,5)}", flush=True)
COMBOS = {
    "CUR+ALL": ["CUR37", "ALL49"],
    "CUR+KNN": ["CUR37", "KNN41"],
    "ALL+KNN": ["ALL49", "KNN41"],
    "BAG3": ["CUR37", "ALL49", "KNN41"],
}
for tag, parts in COMBOS.items():
    per = [roc_auc_score(y, np.mean([RES[p][i] for p in parts], axis=0)) for i in range(len(SEEDS))]
    m = float(np.mean(per))
    flag = "   <-- beats champion" if m > CHAMPION + EPS else ""
    print(f"  {tag:8s} {m:.5f} {np.round(per,5)}{flag}", flush=True)
    if m > best[1]:
        best = (tag, m, parts)

win, wm, wparts = best
print(f"\n>>> best = {win}  {wm:.5f}  parts={wparts}")
print(f"    (champion single-set CUR37 = {np.mean([roc_auc_score(y, p) for p in RES['CUR37']]):.5f})")

if wm > CHAMPION + EPS:
    print("\n=== shipping the bagged meta ===", flush=True)
    Tlist = []
    for p in wparts:
        G, Gt = GALL, GTALL
        idx = [names.index(k) for k in SETS[p]]
        gp, gtp = GALL[:, idx], GTALL[:, idx]
        (A1, B1), (A2, B2) = design(SETS[p], gp, gtp)
        pc = PCA(n_components=None, random_state=0).fit(gp)
        A3, B3 = np.column_stack([gp, pc.transform(gp)]), np.column_stack([gtp, pc.transform(gtp)])
        for (A, B) in ((A1, B1), (A2, B2), (A3, B3)):
            for s in (42, 777):
                p2 = dict(LT, random_state=s, bagging_seed=1000 + s, feature_fraction_seed=2000 + s)
                m = lgb.train(p2, lgb.Dataset(A, y), 400, callbacks=[lgb.log_evaluation(0)])
                Tlist.append(gauss(m.predict(B)))
        print(f"  fit set {p} (3 parts x2 seeds) ({time.time()-t0:.0f}s)", flush=True)
    pred = np.mean(Tlist, axis=0)
    out = "submission_meta_bagset.csv"
    s2 = pd.read_csv("数据/test.csv")[["id"]].copy()
    s2["Will_Buy_EV"] = rankdata(pred) / rankdata(pred).max()
    s2.to_csv(out, index=False)
    prev = pd.read_csv("submission_meta_basis.csv")["Will_Buy_EV"]
    print(f"  wrote {out}  n={len(s2)}  spearman vs champion = "
          f"{s2['Will_Buy_EV'].corr(prev, method='spearman'):.5f}")
else:
    print(f"\n>>> bag does not beat the champion by >{EPS:.0e}; "
          f"submission_meta_basis.csv (0.94320) stands. No file written.")
print(f"total {time.time()-t0:.0f}s")
