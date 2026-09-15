# -*- coding: utf-8 -*-
"""
LB 结果分析（只读，不改动任何模型/提交文件）
目的：
 1) CV -> LB 偏移表：哪些族（单模 / 线性融合 / 非线性元）的 CV 会高估 LB
 2) 公开榜噪声校准：把 OOF 当"候选预测"，随机抽 20%/30%/75% 行重算 AUC，
    得到"两个高度相关候选在同一子集上的 AUC 差"的抽样标准差 sigma。
    有了 sigma，就能判断实测的 LB 差（如 -0.00040）是信号还是噪声。
 3) 已提交文件之间的 spearman（冗余度）
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(2024)

# ---------- 1) CV -> LB ----------
# CV 取值 = 本项目 cross-fit OOF 记录值；LB = 用户回报的公开榜分数
T = [
    ("submission_meta_bagset.csv",   0.94323, 0.94238, "非线性元 12 模型秩平均(全成员集bagging)"),
    ("submission_meta_basis.csv",    0.94320, 0.94023, "非线性元 换基(pwdiff+3raw+PCA)"),
    ("submission_final_stack.csv",   0.94275, 0.94278, "线性元 37 成员"),
    ("submission_final_lr_C1.0.csv", 0.94246, 0.94216, "线性元 较早版本"),
    ("submission_ensemble.csv",      0.94222, 0.94148, "早期加权融合"),
    ("submission_lgb_blend2.csv",    0.94220, 0.94187, "早期 greedy 融合"),
    ("submission_lgb_fe2.csv",       0.94203, 0.94181, "单模 LGB(FE2)"),
    ("submission_lgb_blend.csv",     0.94200, 0.94185, "早期融合"),
    ("submission_lgb_hpo.csv",       0.94188, 0.94165, "单模 LGB(HPO)"),
    ("submission_lgb_fe.csv",        0.94180, 0.94123, "单模 LGB(FE)"),
    ("submission_lgb_baseline.csv",  0.94150, 0.94156, "单模 LGB 基线"),
]
print("=" * 96)
print("CV -> LB 偏移表（按 LB 降序）")
print("=" * 96)
print(f"{'文件':32s} {'CV(cross-fit)':>13s} {'LB(public)':>11s} {'LB-CV':>9s}  说明")
FAM = {"非线性元": [], "线性元/融合": [], "单模": []}
for f, cv, lb, note in sorted(T, key=lambda r: -r[2]):
    d = lb - cv
    print(f"{f:32s} {cv:>13.5f} {lb:>11.5f} {d:>+9.5f}  {note}")
    if "非线性元" in note:
        FAM["非线性元"].append(d)
    elif "单模" in note:
        FAM["单模"].append(d)
    else:
        FAM["线性元/融合"].append(d)
print()
for k, v in FAM.items():
    if v:
        print(f"  {k:8s} n={len(v)}  平均偏移 {np.mean(v):+.5f}  范围 [{min(v):+.5f}, {max(v):+.5f}]")

best_lb = max(T, key=lambda r: r[2])
old_best = 0.94185
print(f"\n>>> 实测最佳 LB = {best_lb[1]:.5f}  ({best_lb[0]})")
print(f"    对比你此前最好 {old_best:.5f}  =>  {best_lb[2]-old_best:+.5f}")

# ---------- 2) 公开榜噪声校准 ----------
print("\n" + "=" * 96)
print("公开榜噪声校准（用 OOF 预测做代理，随机子集重算 AUC）")
print("=" * 96)
tr = pd.read_csv("数据/train.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
n = len(y)

# 两个"同一成员池、只差组合器形态"的候选（正好对应我们要判定的那对提交）
pairs = [
    ("叶内线性树元 T2", "oof_meta_T2.npy", "线性元 L", "oof_meta_L.npy"),
]
for na, fa, nb, fb in pairs:
    A, B = np.load(fa), np.load(fb)
    a_full, b_full = roc_auc_score(y, A), roc_auc_score(y, B)
    print(f"\n候选A = {na}  (OOF AUC {a_full:.5f})")
    print(f"候选B = {nb}  (OOF AUC {b_full:.5f})")
    print(f"  两者在全部 OOF 上的配对差 A-B = {a_full-b_full:+.5f}")
    print(f"  {'子集比例':>8s} {'行数':>9s} {'SD(AUC_A)':>11s} {'SD(AUC_B)':>11s} "
          f"{'SD(A-B) 配对':>14s} {'|实测-0.00040| 相当于':>22s}")
    for frac in (0.20, 0.30, 0.75):
        m = int(frac * n)
        da, db = [], []
        for _ in range(150):
            idx = rng.choice(n, m, replace=False)
            da.append(roc_auc_score(y[idx], A[idx]))
            db.append(roc_auc_score(y[idx], B[idx]))
        da, db = np.array(da), np.array(db)
        d = da - db
        sd_pair = d.std(ddof=1)
        z = 0.00040 / sd_pair
        print(f"  {frac:>8.0%} {m:>9d} {da.std(ddof=1):>11.5f} {db.std(ddof=1):>11.5f} "
              f"{sd_pair:>14.5f} {z:>18.2f} sigma")
    print("  注：SD(AUC_A/B) 是单份提交自身的榜分波动（会盖住真实进步）；"
          "SD(A-B) 是配对差波动（同一测试集上两候选的相对判定）")

# ---------- 3) 已提交文件的冗余度 ----------
print("\n" + "=" * 96)
print("已提交文件之间的 spearman（>0.999 视为实质同一份）")
print("=" * 96)
V = {}
for f, _, _, _ in T:
    try:
        V[f] = pd.read_csv(f)["Will_Buy_EV"].values
    except Exception as e:
        print("  读取失败", f, e)
names = list(V)
print(f"{'':34s}" + "".join(f"{i:>7d}" for i in range(len(names))))
for i, a in enumerate(names):
    row = "".join(
        f"{pd.Series(V[a]).corr(pd.Series(V[b]), method='spearman'):>7.4f}" for b in names
    )
    print(f"{i:2d} {a:31s}{row}")
