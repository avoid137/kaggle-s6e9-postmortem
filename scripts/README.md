# scripts/

Research code, published for provenance. Reading order below; each script is self-contained and writes `.npy` / `.csv` caches into the working directory.

**Expected inputs**: the competition files at `data/train.csv` and `data/test.csv` (Kaggle *Playground S6E9 — Predicting Electric Vehicle Purchases*). Paths are hard-coded relative to the repo root; there is no CLI.

| script | what it does | output |
|---|---|---|
| `nb_feature_stats.py` | §1 of the write-up: per-feature univariate AUC (raw for numerics, **out-of-fold** target encoding for categoricals), the greedy forward build-up curve, and dataset stats | `data/feature_auc.csv`, `data/cumulative.csv`, `data/stats.json` |
| `lb_vs_cv.py` | CV ↔ LB offset table for every submission, plus the first cut at leaderboard-noise calibration | console |
| `lb_noise2.py` | the *paired*-difference noise calibration: resample rows, recompute AUC, compare SD of one submission vs SD of A−B. **This is the measurement that turns a −0.00040 gap into ≈8σ** | console |
| `stack_final3.py` | the **37-member linear stack** — the actual best submission (LB 0.94278). Curated member set pinned by explicit name (see pitfall 2 in the log) | `submission_final_stack.csv` |
| `stack_meta9.py` | the **non-linear meta** + member-set bagging. Best CV of the project (0.94323) and the biggest LB regression (0.94238) — the subject of §5 | `submission_meta_bagset.csv` |
| `meta_transfer_diag.py` | the smoothness diagnostic: shrink OOF member columns toward the row consensus until their mean pairwise \|corr\| matches the measured test-side value (λ=0.12), then re-run cross-fit. It **passed** (0.94314 → 0.94313) and the leaderboard still disagreed → see the caveat in §6 | console |
| `verify_s3_weights.py` | leave-one-out marginal value of every member in the corrected (gaussianised-rank) protocol — the honest replacement for the invalidated "blend weight = 0" claims of §4 | `s3_weight_check.csv` |
| `make_notebook.py` | regenerates `notebooks/EV_S6E9_postmortem.ipynb` from inlined data (no dataset needed) | the notebook |

## Not included

The ~50 scripts that generated individual members (`weak_diverse.py`, `diverse_more*.py`, `lgb_*.py`, `cat_model.py`, …) and the `.npy` member cache — they are mechanical variations on the same template and the cache is several hundred MB. The write-up's §3 table records every result.

## Rebuilding from scratch

```bash
python nb_feature_stats.py           # needs data/train.csv
python stack_final3.py               # needs the member cache (oof_*.npy / test_*.npy)
python verify_s3_weights.py
```

To regenerate the notebook alone, no data is required:

```bash
python make_notebook.py
```
