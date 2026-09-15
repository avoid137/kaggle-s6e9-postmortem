# -*- coding: utf-8 -*-
"""
为公开 notebook 生成"真实可引用"的数据：
 1) 13 个特征的**单变量 AUC**（数值特征取方向修正后的原始 AUC；分类特征用 5 折 OOF 目标编码）
 2) 数值特征的**分箱 OOF 目标编码 AUC**（捕捉非单调关系）
 3) **贪心累积曲线**：按单变量 AUC 降序逐个加入特征，LGB 3 折 CV 的真实 AUC
    （这是"信号高度集中"那张图的实测来源）
 4) 若干描述统计（正例率、Subsidy 硬门率、缺失值）
输出到 notebook_data/
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

os.makedirs("notebook_data", exist_ok=True)

tr = pd.read_csv("数据/train.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n = len(y)

NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
       "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
       "Environmental_Concern_Level"]
CAT = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
       "Subsidy_Available", "Range_Anxiety_Level"]

skf = StratifiedKFold(5, shuffle=True, random_state=2024)


def oof_te(codes):
    enc = np.zeros(len(y))
    for tri, vai in skf.split(codes, y):
        m = pd.Series(y[tri]).groupby(codes[tri]).mean()
        enc[vai] = pd.Series(codes[vai]).map(m).fillna(y[tri].mean()).values
    return enc


rows = []
for c in NUM:
    x = tr[c].values.astype(float)
    a = roc_auc_score(y, x)
    a = max(a, 1.0 - a)                      # 方向修正：取更强的那个方向
    q = pd.qcut(tr[c], 20, duplicates="drop").cat.codes.values
    ab = roc_auc_score(y, oof_te(q))          # 分箱后允许非单调
    rows.append(dict(feature=c, kind="numeric", univariate_auc=a, binned_oof_auc=ab))
for c in CAT:
    code = pd.factorize(tr[c])[0]
    ab = roc_auc_score(y, oof_te(code))
    rows.append(dict(feature=c, kind="categorical", univariate_auc=np.nan, binned_oof_auc=ab))

F = pd.DataFrame(rows).sort_values("binned_oof_auc", ascending=False).reset_index(drop=True)
F.to_csv("notebook_data/feature_auc.csv", index=False)
print("=== 单变量 AUC（按分箱 OOF 降序）===")
print(F.to_string(index=False))

# ---------- 贪心累积曲线 ----------
rng = np.random.default_rng(0)
sub = rng.choice(n, 200_000, replace=False)
Xs = tr.iloc[sub].reset_index(drop=True)
ys = y[sub]
for c in CAT:
    Xs[c] = Xs[c].astype("category")

order = F["feature"].tolist()
P = dict(objective="binary", metric="auc", learning_rate=0.06, num_leaves=31,
         feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
         min_child_samples=100, verbose=-1, num_threads=20, seed=1)

print("\n=== 贪心累积曲线（200k 子样本, 3 折, 400 轮）===")
curve = []
for k in range(1, 14):
    cols = order[:k]
    cats = [c for c in cols if c in CAT]
    oof = np.zeros(len(ys))
    for tri, vai in StratifiedKFold(3, shuffle=True, random_state=7).split(Xs[cols], ys):
        d = lgb.Dataset(Xs[cols].iloc[tri], ys[tri], categorical_feature=cats)
        m = lgb.train(P, d, 400)
        oof[vai] = m.predict(Xs[cols].iloc[vai])
    auc = roc_auc_score(ys, oof)
    curve.append(dict(k=k, added=cols[-1], auc=auc))
    print(f"  k={k:2d}  +{cols[-1]:28s} AUC={auc:.5f}", flush=True)
pd.DataFrame(curve).to_csv("notebook_data/cumulative.csv", index=False)

# ---------- 描述统计 ----------
sub_rate = tr.groupby("Subsidy_Available")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean())
hcp_rate = tr.groupby("Home_Charging_Possible")["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean())
stats = dict(
    n_train=int(n),
    n_test=int(len(pd.read_csv("数据/test.csv", usecols=["id"]))),
    pos_rate=float(y.mean()),
    n_features=len(NUM) + len(CAT),
    missing=int(tr.isna().sum().sum()),
    dup_rows=int(tr.drop(columns=["id"]).duplicated().sum()),
    subsidy_rate={k: float(v) for k, v in sub_rate.items()},
    home_charge_rate={k: float(v) for k, v in hcp_rate.items()},
    cardinality={c: int(tr[c].nunique()) for c in CAT},
)
json.dump(stats, open("notebook_data/stats.json", "w"), indent=2, ensure_ascii=False)
print("\n=== 描述统计 ===")
print(json.dumps(stats, indent=2, ensure_ascii=False))
print("\nDONE")
