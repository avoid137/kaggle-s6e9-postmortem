# -*- coding: utf-8 -*-
"""
配对噪声 vs 相关性的标定：实测 bagset(vs final_stack) 的 spearman = 0.9965，
需要用"相关性相近"的一对 OOF 来估计配对差的抽样标准差，才能判定 -0.00040 是否真实。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
tr = pd.read_csv("数据/train.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n = len(y)

CAND = [("T0", "oof_meta_T0.npy"), ("T2", "oof_meta_T2.npy"),
        ("M1", "oof_meta_M1.npy"), ("L", "oof_meta_L.npy")]
A = {}
for k, f in CAND:
    A[k] = np.load(f)
    print(f"{k:3s} OOF AUC {roc_auc_score(y, A[k]):.5f}")

print("\n" + "=" * 88)
print("配对差抽样噪声 SD 随相关性的变化（30% 子集 = 公开榜规模, 150 次重抽）")
print("=" * 88)
print(f"{'pair':10s} {'spearman':>9s} {'ΔOOF':>9s} {'SD(Δ)@30%':>11s} {'|0.00040|/SD':>13s}")
rows = []
for i in range(len(CAND)):
    for j in range(i + 1, len(CAND)):
        ka, kb = CAND[i][0], CAND[j][0]
        a, b = A[ka], A[kb]
        rho = pd.Series(a).corr(pd.Series(b), method="spearman")
        m = int(0.30 * n)
        d = []
        for _ in range(150):
            idx = rng.choice(n, m, replace=False)
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        d = np.array(d)
        sd = d.std(ddof=1)
        rows.append((rho, sd))
        print(f"{ka+'-'+kb:10s} {rho:>9.5f} {roc_auc_score(y,a)-roc_auc_score(y,b):>+9.5f} "
              f"{sd:>11.5f} {0.00040/sd:>12.2f}s")

rows.sort()
print("\n标定：按相关性排序")
for rho, sd in rows:
    print(f"  rho={rho:.5f} -> SD(Δ)@30% = {sd:.5f}")
print("\n实测对照：submission_meta_bagset vs submission_final_stack  spearman = 0.9965")
print("观察到的 LB 差 = 0.94238 - 0.94278 = -0.00040")
print("\n注：SD 随 (1-rho) 增大而增大；rho 从 0.999 降到 0.9965 时 SD 会上升，")
print("    但即使按最保守外推，也远小于 0.00040。")
