# Kaggle 发布文案（notebook + Discussion）

配套产物：`notebooks/EV_S6E9_postmortem.ipynb`。

## 0. 发布状态（2026-09-15 已线上核对）

已发布：<https://www.kaggle.com/code/avid147/s6e9-postmortem-cv-up-lb-down-49-models>

用公开 API（`/api/v1/kernels/pull`、`/api/v1/kernels/output`，无需登录）逐字段核对的结果：

| 检查项 | 线上实际值 | 判定 |
|---|---|---|
| Title | `S6E9 postmortem: CV up, LB down (49 models)`（43 字符） | ✅ 与 §1 推荐一致 |
| 可见性 | `isPrivate = false`（匿名可访问） | ✅ Public |
| 竞赛归属 | `competitionDataSources = ['playground-series-s6e9']`，页面挂 "Competition Notebook" | ✅ 进竞赛 Code 页 → 满足规则 8.B |
| 数据源 | 仅挂 Playground S6E9（无外部 dataset / kernel） | ✅ |
| 运行 | 日志完整、无 error，`nbconvert` 写出 699,801 B notebook + 5 个 `__results___files/` 目录 | ✅ 5 张图随版本保存 |
| 内容版本 | 与仓库 commit `4011e3b` 一致（已含 "I have no idea how they did it" 缓和措辞、无 "42 models"、无 "very unlikely"） | ✅ 措辞正确 |
| **Tags** | `categoryIds = []` | ❌ **未填** —— 影响 Code 页搜索/筛选曝光 |
| **License** | 页面显示 **Apache 2.0** | ⚠️ 与仓库 `LICENSE` 的 **MIT** 不一致（两者都是 OSI 许可，不影响规则 8.B，但建议统一） |
| 版本 | `currentVersionNumber = 1` | ⚠️ 该版本早于 18:00 的钩子段提交（`0d47ed5`），差 2 行 |

### notebook 的合规核对（逐条对规则，别只凭"感觉没问题"）

| 规则 | 要求 | notebook 的实际情况 | 判定 |
|---|---|---|---|
| **7.B** Data Security | 不得把 Competition Data 发布/再分发 | notebook 里**没有任何数据行**：所有数字都是内联的派生聚合值；唯一的读数据单元格是 Appendix（cell 19），它 `read_csv("/kaggle/input/.../train.csv")` 只为**本地重算**，输出仅一张 13 行的 `feature / type / univariate_auc`（4 位小数）汇总表 | ✅ 只发聚合统计 |
| **8.A** Private Code Sharing | 比赛期间不得**私下**共享代码 | 发布形态是 Public notebook + 公开 GitHub 仓库，没有私信/邮件/私聊群分享 | ✅ 走的是公开 |
| **8.B** Public Code Sharing | 公开分享需 ① OSI 且不限商用的许可 ② 发在本场 Discussion / Notebook | ① Kaggle notebook 页显示 Apache 2.0（OSI、不限商用）；② 已挂上 S6E9 成为 Competition Notebook，出现在本场 Code 页 | ✅ 两条都满足 |
| 讨论区规范 | 不得拉票 / 自我推广 | notebook 与帖文里**没有任何**求赞字样；只在正文里放自己的 notebook 链接（分享，不是拉票） | ✅ |
| **7.C** External Data | 外部数据须公开免费 | 只用本场竞赛数据，无外部数据 | ✅ |

- 唯一要长期守住的细节：**Appendix 只准输出聚合值**。一旦有人往里加 `tr.head()`、`value_counts()` 之类的原始行输出，就从"合规"滑到"发布竞赛数据"。
- License 不一致（Kaggle Apache 2.0 / 仓库 MIT）**不是合规问题**——两者都是 OSI 且都不限商用，8.B 只看这一点；统一只是为了对外一致。
- 顺带一提：S6E9 是 Playground（提交文件），不是 Code Competition，所以 notebook 的 `enableInternet: true` 也没有任何限制问题。

**剩下的三个动作**（都在 notebook 页面右侧边栏/头部，1 分钟）：

1. 补 **Tags**：`tabular` · `stacking` · `ensemble` · `cross-validation` · `lightgbm`（弹窗是**点选**不是手打；中文界面显示中文名，如"表格""模型对比""优化""初学者""LightGBM"）
2. License 下拉改成 **MIT**（与 GitHub 一致）—— 或反过来把仓库 LICENSE 换成 Apache-2.0，二选一即可
3. （可选）把本地 `notebooks/EV_S6E9_postmortem.ipynb` 重新 **Import → Save Version**，得到 v2，带上 cell 1 的钩子段和 Closing 里的仓库链接

### Discussion 帖（已发布 2026-09-15 18:57）

线上：<https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741447>（作者显示名 `Dingjunpeng`，匿名可访问）

| 检查项 | 线上实际值 | 判定 |
|---|---|---|
| 标题 | `[Postmortem] My CV went up 0.0005 and my LB went down 0.0005 — same members, same folds`（87 字符） | ✅ 与 §3 一致 |
| 正文 | 与 §3 草稿逐段一致（项目符号未被吞、加粗斜体正常） | ✅ |
| notebook 链接 | 已渲染为可点链接，指向 `.../code/avid147/s6e9-postmortem-cv-up-lb-down-49-models` | ✅ |
| 求赞字样 | 无（只放自己的 notebook 链接 = 分享 ≠ 拉票） | ✅ |
| Tags / 票数 | 站外抓不到（讨论页是客户端渲染，`urllib` 只拿到 5.6 KB 壳页；渲染抓取不返回 tag 胶囊） | ⚠️ 需**在自己页面上目视确认**标题下方是否有标签胶囊 |
| 格式 | `...than speculate. To be clear...` 线上显示为 `...than speculate.To be clear...`（富文本粘贴吃掉句号后的空格） | ⚠️ 点 **Edit** 在句号后补一个空格或分段即可 |

> **`markdown 源码里对 ≠ 线上渲染后还对`**：粘贴到富文本编辑器后，回到线上页面把标题、句间空格、列表层级、链接可点性通读一遍再收工——这类小瑕疵出现在"可信度就是全部卖点"的复盘帖上，代价不成比例。

## ⚠️ 发布前必做（这三条决定成败）

1. **必须把 S6E9 的竞赛数据挂为 notebook 的输入** —— 否则它不会出现在竞赛的 Code 页，**也就不参与 Code 奖牌评选**。notebook 本身不需要数据也能跑（所有数字已内联），挂输入只是为了让它被归到该竞赛下。
2. **标题上限 50 字符**（见 §1），且 **notebook 没有 Description 字段**（见 §2）—— 钩子只能放标题 / 第一个 markdown 单元格 / Discussion 正文。
3. **保存版本时别丢输出**：`Save & Run All (Commit)` 默认保存图表；若用 **Quick Save**，要先把 **Save Output** 设成 `Always save output` / `Save output for this version`，否则别人 fork 看不到图。
4. **不要拉票。** Kaggle 明令：*"Do not self-promote your Notebook by posting links to it across Kaggle or by asking other users to upvote your work."* 分享链接、介绍内容都可以，**请求投票不行** —— 互赞群会被封号。奖牌只能靠自然投票。

> 另注（**官方原文**，取自 `kaggle.com/progression/code`）：*"Not all upvotes count towards medals. Only votes from users who are **Expert Tier or higher** count for medals (self-votes never count). In addition, we use secret algorithms to discount some votes to prevent abuse, as a result the total required will often be higher than the amounts listed below."* → 铜牌标称 5 票，实际往往需要 **10–15 个自然赞**（银 20 / 金 50）。
>
> Discussion 通道的**计票口径官方抓不到原文**（`progression/discussions` 是纯客户端渲染），社区多来源一致的说法是"**Novice 用户的票不计**"（比 Code 的 Expert+ 松）。保守按与 Code 同标准估算，不要把"1 个净赞"当稳定收益。

---

## 1. Notebook 标题（粘贴到 Title）

> ⚠️ **Kaggle 硬限制：Title 只能填 5–50 个字符。** 输入框自带 "x / 50" 计数器，超过直接无法输入。
> 所以原来拟的 93 字符长标题**填不进去** —— 那句话现在的位置是 notebook 第一个 markdown 单元格的 H1（那里不限长，已内置）。

**推荐（43 字符）**

```
S6E9 postmortem: CV up, LB down (49 models)
```

**备选（都在 50 以内）**

```
S6E9 postmortem: 49 models, one silent bug      # 42
S6E9: a CV gain that didn't survive the LB      # 42
CV up, LB down: a S6E9 postmortem (49 models)   # 45
```

- 保留 `S6E9` 是为了在 Code 页被搜到；**Code 列表卡片只显示 标题 / 作者 / 票数 / 更新时间，没有摘要**，所以钩子本身必须塞进标题（尽量放在前 ~40 字符）。
- slug 由标题推导 —— 实际发布后：`s6e9-postmortem-cv-up-lb-down-49-models`，kernel `id = 134472821`，作者名 `avid147`（显示名 Dingjunpeng）。**改名会改 slug，链接不失效但 `id` 不变**。

## 2. 「描述」放哪里 —— ⚠️ Kaggle notebook **没有 Description 字段**

**没有地方粘。** 已核对 Kaggle 当前前端代码：编辑器、Save Version 弹窗、notebook 页面侧边栏都**不存在** notebook 的 description / subtitle 输入框（页面上有 Description 输入框的是 dataset、benchmark task、竞赛提交面板，都不是 notebook）。页面头部那个副标题只在 Learn 简化界面里显示 *lesson* 的描述，与 notebook 无关。

真正可填的 notebook 字段只有：**Title（≤50 字符）· Tags · License · Input**。因此下面这段钩子文案改放三处：

1. **notebook 第一个 markdown 单元格** —— 点开页面第一眼看到的就是它（原文已内置于 `notebooks/EV_S6E9_postmortem.ipynb` 的 cell 1）。
2. **Discussion 帖标题 + 正文首段**（§3 草稿已含同样开头）。
3. **Title**（见 §1，50 字符内）。

下面这段仍保留，作为 Discussion 正文 / notebook 开头的候选措辞：

```markdown
I pushed cross-validated AUC from 0.94275 to 0.94323 (+0.00048) with a fancier combiner, and my public score *dropped* to 0.94238. Here is why — plus 17 "obvious" improvements I falsified with measurements along the way.

**TL;DR**

- The signal is absurdly concentrated: one feature (`Environmental_Concern_Level`) gives AUC 0.8435 alone; 3 features reach 0.9354; **CV peaks at 7 of 13 features**. Roughly 96% of the achievable gain lives in 3 columns.
- I **falsified 17** plausible upgrades (MLP/RealMLP, monotone constraints, DART, 10-fold + full refits, rank averaging, CatBoost CTR, extra datasets, pseudo-labelling, k-NN variants, RBF kernels...) — each with a measured number, not an opinion.
- A **normalisation bug** silently invalidated a whole round of my "this model contributes nothing (weight 0)" conclusions. If you have ever drawn that conclusion from a blend weight, that section is for you.
- **The main result:** a global linear combiner scored 0.94275 CV / **0.94278 LB**. A tree / leaf-linear combiner scored **0.94323 CV** / 0.94238 LB. A **0.00088 swing in the wrong direction**, and statistically solid at **≈8σ** once leaderboard noise is calibrated as a *paired* difference (paired SD ≈ 5e-5 vs single-submission SD ≈ 5e-4).
- **Root cause:** the members' OOF columns were each **one** model's prediction; the test columns were **5-fold averages**. Different *aggregation order* → the combiner fits corrections calibrated to OOF-specific noise that is averaged away at inference. **Linear combiners are structurally immune; expressive ones are not.**

**No dataset needed, no training.** Every number is inlined, so you can fork this and re-plot it in about 3 seconds.

The part I'd most like to be wrong about is the root cause — if you have a counter-example where a non-linear stack *did* transfer, please say so in the comments.
```

**notebook 侧边栏真正要填的三样**（打开 notebook 页 → 右侧边栏；窄屏先点 "View Metadata" 展开）：

| 字段 | 填什么 |
|---|---|
| **Tags** | `tabular` · `stacking` · `ensemble` · `cross-validation` · `lightgbm` |
| **License** | MIT（与 GitHub 仓库一致） |
| **Input** | 必须挂 **Playground S6E9** 竞赛数据（否则不进竞赛 Code 页、不参与 Code 奖牌） |

## 3. Discussion 帖（另一条奖牌通道，门槛低得多）

Code 奖牌要 5 票，**Discussion 铜牌只要 1 个净赞**，值得一起做。发帖链接自己的 notebook 是允许的（分享 ≠ 拉票），但**同样不要出现"请点赞/求 upvote"字样**。

**发帖位置**：竞赛页 → `Discussion` 标签 → 右上/顶部 **New Topic** → 那一页就是（`TOPIC TITLE` 输入框 + 富文本正文 + `Add Tags` + `Publish Topic`）。

**这个编辑器的几个实测事实**（已核对 Kaggle 当前前端 bundle，别再凭印象写发布说明）：

| 事实 | 影响 |
|---|---|
| 正文编辑器是**富文本**（`showCharacterCount`、`useMentions`、存储走 markdown），toolbar 里有**表格按钮**（截图里第 11 个图标） | 粘 markdown 表格**可能**不自动渲染 → 正文默认用**项目符号**，想用表格就用 toolbar 的表格按钮手插 3×3 |
| `allowExternalImages:false` | **不能**用外链贴图，要插图必须上传本地文件（我们不需要图，图文在 notebook 里） |
| `Topic Title` 字段**没有 maxLength**（只有正文有字数计数） | 标题可以长，但列表页会截断 → 控制在 **90 字符内** |
| `Add Tags` 是**弹窗选择器**（有分区、有 maxTags），不是自由输入 | **上限 5 个、下限 0 个（可有可不有）**，提示语是 *"Apply up to 5 tags to help Kaggle users find your content."*；分类固定为 `SUBJECT / AUDIENCE / TECHNIQUES / PACKAGES / DATA_TYPE / GEOGRAPHY`，**不能自己造标签** → 从候选中点选贴合的就够了；tag **不是奖牌条件**（奖牌只看票数） |

**标题**（推荐 87 字符：保留 "CV 涨 / LB 跌" 的钩子，并补上"同样的成员、同样的折"这个可信度信号）

```
[Postmortem] My CV went up 0.0005 and my LB went down 0.0005 — same members, same folds
```

**更短的备选（68 字符，列表页截断更安全）**

```
[Postmortem] CV up 0.0005, LB down 0.0005 — same members, same folds
```

**正文（直接整段复制；已按"列表页只显示前两行"优化，第一句就是结论）**

```markdown
I changed one thing — the combiner — and my cross-validated AUC went up while my public score went down. Same member pool, same folds, same features. I'd rather write the negative result down than bury it.

Notebook (every number inlined, no data needed, runs in ~3 s): <notebook link>

Three combiners over the same OOF matrix:

- global linear, 37 curated members — CV 0.94275 / public LB 0.94278
- leaf-linear tree + basis augmentation (pairwise diffs, PCA) — CV 0.94320 / public LB 0.94023
- rank-average of several such metas — CV 0.94323 / public LB 0.94238

The CV gain was clean and monotone with combiner capacity, the ablations were self-consistent, monotone constraints *hurt* (so it wasn't noise-smoothing in disguise), and a smoothness diagnostic passed. It still lost ~0.0004 on the leaderboard.

On "isn't 0.0004 just noise?" — the SD of a single submission is ≈5e-4, but the SD of the *paired* difference between two highly correlated submissions (ρ≈0.996) is ≈5e-5. Against the paired SD, -0.0004 is ≈8σ. Calibrating the paired difference instead of the single-submission SD is a one-liner, and it flipped my conclusion. Worth doing before you write off any gap on a dense leaderboard.

My explanation, and the part I'm least sure about: the member OOF columns are each the prediction of **one** model that never saw the row, while the test columns are **averages of K=5** fold models. So a tree can carve out splits like "member A is unusually high relative to member B" whose firing pattern depends on the column's noise level; fit at σ, deploy at σ/√K, and those corrections land on the wrong rows. A global linear combiner only estimates first moments, and is structurally immune.

If that's the mechanism, the fix is to change the *representation*, not the combiner: repeated K-fold across seeds, averaging each row's out-of-fold predictions over replicates so both sides are averages of K models. I did not run that — I'm reporting the diagnosis, not the cure.

Two things I'd value your take on:
1. A counter-example where a non-linear stack *did* transfer, with matched aggregation order?
2. A cleaner test than mine — I shrank the OOF columns toward the row consensus until their mean correlation matched the test-side value. It passed, so it clearly wasn't sufficient.

Scope note, because it matters: I'm reporting that *I* couldn't make this work, not that it can't be done. There are people above me on this leaderboard who clearly closed this gap, and I don't know what they did differently — I'd rather say that plainly than speculate. To be clear, I'm not fishing for anyone's approach while the competition is still running — what would help is a yes/no on whether the mechanism above is wrong, and that's answerable without revealing anything.

The notebook also has the 17 approaches I falsified, each with a measured number rather than an opinion. If you're stuck around 0.943, that table is probably the most useful part.
```

- 正文里的 `<notebook link>` 替换成真实 URL：`https://www.kaggle.com/code/avid147/s6e9-postmortem-cv-up-lb-down-49-models`
- 正文用**项目符号**而不是表格，是为了避开富文本编辑器对粘贴 markdown 表格的不确定性；想上表格就用 toolbar 的表格按钮手插。

**"比赛还没结束，这样发帖/这样提问算不算求答案？"—— 边界在哪**

| 允许 | 违规 |
|---|---|
| 公开讨论方法、公开提问、公开分享自己的代码与负结果 —— 规则 **8.B** 明确允许 public sharing，且讨论区本身就是干这个的（编辑器占位文字是 Kaggle 自己写的 *"Stuck on a problem, or have a question about the competition?"*） | **要求或接受私下分享**（私信 / 邮件 / 私聊群）别人的代码或方案 —— 对方触犯 **7.B / 8.A**，可能连带双方被取消资格 |
| 问"我这个机制解释对不对"——这是可证伪的方法论问题 | 问"你的 trick 是什么 / 能不能把方案发我"——低质量搭便车，容易招负评 |
| 比赛结束后再问"你们怎么做的"（那是常态，也是收尾传统） | 任何形式的**拉票**（求 upvote）—— 会被封号 |

三处我们做对的细节（可以照抄这个套路）：

1. **问在公开帖里，绝不去私信** —— 公开帖里的回答属于 8.B 允许的 public sharing，对方零风险。
2. **只问机制对错，不问方案细节** —— 一句 "yes/no on whether the mechanism is wrong" 完全不需要对方透露任何东西。
3. **先给后问** —— 我们公开了 17 条否定结果 + 全部数字 + 完整脚本，"给予"远大于"索取"，因此不会被读成搭便车。

> 另外，"There are people above me…I don't know what they did differently" 这句只是**陈述自己的无知**，没有向任何人提出请求 —— 它本身完全不构成索要，是安全的。真正需要防的是有人看到它之后主动**私信**你讲方案：**收到了也别用**，回复"请发在公开帖里"即可。

### 3.1 回帖记录（2026-09-16，前 300 名选手公开回复）

对方直接回答了那句 yes/no：**"From here it is no, it is not wrong."** —— 即**机制没有被证伪**，并给了独立证据：

| 他的实验 | 他的数字 | 对我们的意义 |
|---|---|---|
| 把第二个 fold split 平均进 OOF 估计（每行由 2 个模型打分而非 1 个） | 两条腿分别 **+0.00019 / +0.00006**（差 3 倍） | 不同 member 被扭曲的**幅度不同** —— 这正是"容量高的组合器会 key 住的输入"。他自陈这两条腿不止编码不同，**不能隔离因果** |
| **纯噪声列探针**：往矩阵里插一列纯噪声，看 CV 掉多少 | 单次 OOF 对象 −0.00011；两次运行平均的对象 −0.00007 → 比值 **0.64 ≈ 1/√2** | 同一组合器在**噪声水平不同的同一列**上付出不同代价 → 机制的**受控版**证据（比我们那个"向共识收缩"的平滑诊断强） |
| nested 协议下的组合器增量 | LightGBM stacker **+0.00002** / bagged hill climb **−0.00014** / rank average **+0.00006** | 他的负结论：**即使在 nested CV 下，也没有组合器打败 rank average** |
| 第三方复现（另一个帖子，两个库一致） | **+0.00000 / −0.00008** | 他把复现看得比自己的单次运行更重 → 只当方向性结论 |

**两条边界**（别读多）：

1. 他的两个对象**都是 OOF**（1 模型/行 vs 2 次运行平均），**不是** OOF-vs-test 的对比 → 证明的是机制的**前半段**（噪声水平会改变组合器行为），**没有**替我们证明 OOF→test 那 0.0004 的迁移失败。
2. 他的负结果与我们的**不是同一个**：他说"CV 内部就没有增益"，我们说"CV 有增益但迁移失败"。方向一致，结论不同。

**回复草稿 v2（英文，可直接粘到帖下）**

> v2 相比 v1 增加了两处：① 明确说"OOF→LB 那一段我们俩都没测"是他的边界、也是我的边界，**不是在纠正他**；② 补上一条我自己**还没隔离**的混淆轴（test 列不只是噪声更小，它还是 K 个模型的平均 → **信号本身也可能不同**），并把它作为"matched aggregation order 要匹配的不止噪声水平"的追问。

```
Thank you — that answers the question I asked, and the noise-column probe is a cleaner test than mine. Mine only matched a correlation structure; yours changes the noise level on purpose, and 0.64 against 1/sqrt(2) is the same prediction my mechanism makes: what the combiner pays for a column depends on that column's sigma. Your "the averaged object is two out-of-fold runs, not a leaderboard file" is also exactly the boundary I would draw myself — so between us the mechanism is supported on the OOF side and still untested across OOF -> test.

The asymmetry is the part I did not expect. Averaging a second split moves the two legs very differently (+0.00019 vs +0.00006); if the only change were "less noise", I would have expected a more even shift. So the "change the representation, not the combiner" line in my post should be read as directional, not as a cure — it moves members asymmetrically, which under my own mechanism is another way to land on the wrong rows.

One axis I have not isolated, and I would be interested in your read: my test column is not only less noisy, it is also an average of K models, so its signal is not necessarily the same object as my OOF column either. If that matters, then "matched aggregation order" has to match more than the noise level — otherwise the fix I proposed repairs one mismatch while introducing another. That is answerable with a yes/no and does not require anyone's approach, which is why I am asking it rather than anything about features or losses.

On your stacker deltas: +0.00002 / -0.00014 / +0.00006 all sit in the regime where a paired-difference SD decides whether they mean anything (mine was about 5e-5), so I read your net as "the combiner is worth nothing on this data" rather than "mildly negative" — a stronger claim than my post makes.

And agreed that our two negatives are not the same one: yours is that nothing beats a rank average even nested; mine is that a CV gain did not transfer. Same direction, different finding.
```

**回复草稿 v3（去 AI 味版 —— 主张、数字、段落顺序与 v2 完全一致，只换语气）**

> 起因：用户问"这个应该不会让人感觉太 ai 吧"。自查 v2 发现 8 类典型 LLM 痕迹（见下表），其中最致命的是**结构痕迹**而非用词：5 个段落长度几乎相同、每段都以一句"金句"收尾（`...still untested across OOF -> test` / `...another way to land on the wrong rows` / `...introducing another` / `...a stronger claim than my post makes` / `Same direction, different finding.`），5 个收尾金句排在一起，是人类写不出来的节拍。
>
> v3 只做 6 件事：① 破折号 5 → 0；② 补缩写（`I'd` / `it's` / `didn't` / `don't`），v2 通篇零缩写；③ 让段落长短不均，拆掉"每段一句金句"的节拍；④ 删 `exactly`、`I would be interested in your read` 这类论文腔；⑤ **换掉第 4 段**（见下方说明）；⑥ 结尾用 `Same direction though.` 这种半句收，不写对偶警句。

```
Thanks, that answers what I was actually asking. And your probe is the better test. Mine only matched a correlation structure; you turned the noise level down on purpose, and 0.64 against 1/sqrt(2) is the prediction my mechanism makes too. So I'll take it.

Your "the averaged object is two out-of-fold runs, not a leaderboard file" is the boundary I'd draw as well. Between us the mechanism is supported on the OOF side now. OOF to test is still untested, and I don't think either of us has a clean way to test it, which is probably the honest place to leave it.

What I didn't expect was the asymmetry. Averaging a second split moved the two legs by +0.00019 and +0.00006, and if the only change were less noise I'd have expected something more even. So treat "change the representation, not the combiner" in my post as a direction, not a fix. It moves members by different amounts, and under my own mechanism that is just another way to end up on the wrong rows.

The axis I haven't isolated, and I'm curious what you make of it: my test column is not only less noisy, it's also an average of K models, so its signal may not be the same object as my OOF column at all. If that matters, then "matched aggregation order" has to match more than the noise level, and the fix I proposed would repair one mismatch while introducing another. It's a yes/no question and it doesn't need anyone's approach, which is the only reason I'm asking it.

One last thing, and it's a correction to me rather than to you. I was about to compare your +0.00002 / -0.00014 / +0.00006 against my paired-difference SD of about 5e-5. That one is a leaderboard estimate from paired submissions over the same member set. Yours are nested CV deltas. I never measured the run-to-run noise of my own CV numbers, so I don't really have a ruler for yours, and using mine on yours is the same category error the post is about. No read from me on those three. Your conclusion, that nothing beats a rank average even nested, stands on your numbers and not mine.

On the two negatives, we're still describing different things: yours is that nothing beats a rank average even nested, mine is that a CV gain didn't transfer. Same direction though.
```

**⚠️ v3 第 4 段是有意改的，属于"内容"改动不只是"语气"改动，发布前自己拍板**

- v2 的第 4 段是：*"so I read your net as 'the combiner is worth nothing on this data' rather than 'mildly negative' — a stronger claim than my post makes."* —— 这句拿**我们的 5e-5** 去裁**他的三个 delta**。但我们的 5e-5 是**同一批成员的两次 LB 提交**配对标定出来的；他的三个数是 **nested CV 的 OOF 增量**。**尺子不同、测量的面也不同**（我们从头到尾**没测过自己 CV 数的 run-to-run 噪声**）→ 这正是整篇帖子在讲的那个错。
- 所以 v3 把它改成自我修正：**承认我没有裁他数字的尺子**。信息量没少（读者仍知道"你的 delta 在噪声量级"），但避免了"用我的尺子量你的数据"这种自相矛盾，也让修正落在自己身上而不是他身上（不显得抢话）。
- 如果还是想保留原来那句（他已经在自己的帖里说过 "nothing beats rank average even nested"，所以那句并不冒犯），把 v3 第 4 段换回 v2 版本即可，其余不用动。

**回复草稿 v4（短版，~165 词 / 3 段 —— 用户觉得 v3 太长时用这版）**

> 砍掉的东西：① `Thanks, that answers what I was actually asking` → `Thanks, that answers it`；② 第 3 段的 `and I don't think either of us has a clean way to test it, which is probably the honest place to leave it` 整句删（边界已经说清，不补感慨）；③ `On the two negatives...` 整段删（他本人在留言里已经点出两条负结果不同，**重复他的观察等于没加信息**）；④ `One last thing, and it's a correction to me rather than to you` → `One correction to me, not you`；⑤ 段落 5→3。
>
> 保留的骨架不变：**接受 yes/no + 引他的边界 + 两条自己的新信息（不对称 / test 列同时是 K 模型平均）+ 尺子自我修正**。再短就只能砍第 3 段（尺子那段），会掉到 ~90 词，但代价是丢掉"我不拿自己的尺子量你的数据"这处最有说服力的自我修正 —— 不建议。

```
Thanks, that answers it. And your probe is the better test: 0.64 against 1/sqrt(2) is what my mechanism predicts too, so I'll take it. Your "the averaged object is two out-of-fold runs, not a leaderboard file" is the boundary I'd draw as well, so the OOF side is now supported by both of us and OOF -> test by neither.

Two things from my side. The asymmetry I didn't expect (+0.00019 vs +0.00006), which makes "change the representation, not the combiner" a direction rather than a fix. And one axis I can't check alone: my test column isn't only less noisy, it's also an average of K models, so its signal may not be the same object as my OOF column. If that matters, "matched aggregation order" has to match more than noise. Yes/no is enough, and it needs no one's approach.

One correction to me, not you. I was about to compare your +0.00002 / -0.00014 / +0.00006 against my paired-difference SD of about 5e-5, but that one is a leaderboard estimate and yours are CV deltas. I never measured my own CV noise, so I have no ruler for yours. No read from me.
```

**发布前自查（v4 同样要过）**：破折号 0 个 ✅ / 有缩写（I'd, it's, isn't）✅ / 段落长度不均（1 长 1 长 1 短）✅ / 无连续金句收尾 ✅ / 结尾不再是警句（落在 `No read from me.`）✅

**回复草稿 v5 · FINAL（2026-09-16 19:17 —— ①② 已应用到下方代码块，可直接整段粘贴）**

> 用户自己改的，**实测比 v4 更好**，两处是实质性改进：
> 1. `0.64 against 1/sqrt(2) is what my mechanism predicts too, so I'll take it` → **`is the direction my mechanism predicts, so I read it as consistent rather than confirmatory`** —— 正好落实了我们自己的精度修正（0.707→0.64 折算成绝对量仅约 1e-5，落在单次运行波动量级内 → 只能当**同量级支持**，不是精确验证）。原来那句 `I'll take it` 说过了。
> 2. `OOF -> test is still untested by neither` → **`what neither of us has is a matched-aggregation OOF→test transfer case`** —— 把"没测"具体化成"缺什么"，信息量更高。
>
> **发布前要改三处（①② 已应用，③ 仍可选）**：
> - ✅ **`1/√2` → `1/sqrt(2)`**：他本人原文就写 `1/sqrt(2)`；`√` 在富文本粘贴里有乱码风险（用户粘贴时已变成 `1 / 2` 的分数形式）。**已改**。
> - ✅ **引号 → 直引号 `"`**：原草稿显示为弯引号 `“ ”`，命中 AI 味清单第 18 条。**已改**。
> - ⚪ （可选·合规保险）补回半句 `doesn't need anyone's approach`：**比赛仍在进行中**，这句是"我不是在钓方案细节"的公开凭证。删掉不违规，只少一层防护。若要补，加在第 2 段末：`Nothing in that needs anyone's approach to answer.`
>
> 尺度记录：全篇还剩 **4 处"不是 A 而是 B"的对比句式**（`rather than confirmatory` / `a direction, not a fix` / `on my side, not yours` / `supported by both of us; neither of us has`）。每处都在承担实际区分功能，不构成节拍器问题；若想再降一点，把 `One correction on my side, not yours` 改成 `One correction, on my side` 即少一处对称。

```
Thanks, that answers it. Your probe is the better test: 0.64 vs. 1/sqrt(2) is the direction my mechanism predicts, so I read it as consistent rather than confirmatory. And I'd draw the same boundary: the averaged object is two out-of-fold runs, not a leaderboard file. So the OOF-side mechanism is supported by both of us; what neither of us has is a matched-aggregation OOF->test transfer case.

Two things from my side. The asymmetry I didn't expect (+0.00019 vs +0.00006) makes "change the representation, not the combiner" a direction, not a fix. And one axis I can't check alone: my test column is not only less noisy, it is also an average of K models, so its signal may not be the same object as my OOF column. If that matters, "matched aggregation order" has to match more than noise. Nothing in that needs anyone's approach to answer.

One correction on my side, not yours: I was about to compare your +0.00002 / -0.00014 / +0.00006 against my paired-difference SD of about 5e-5, but mine is a leaderboard estimate and yours are CV deltas. I haven't measured my own CV noise, so I have no ruler for yours. No read from me.
```

- 只做四件事：接受他回答的那句 yes/no / 指出他这组数字在机制上意味着什么 / 明确边界并修正自己"改表示就能修"的说法 / 追一个**不需要透露方案**的 yes-no 问题。
- **不回"求更多细节"、任何地方都不提投票。**
- **自我修正放在回帖里，不改原帖**：原帖那句本来就是条件句（"if that's the mechanism…"），而且公开线程里留下修正痕迹比偷偷 Edit 原帖更站得住 —— 这正是这类负结果复盘最难假装的东西。

### 3.2 第二轮回复（2026-09-17，前 300 名选手更正数字 + 新测量）

他针对 v5 回了两件事：**更正了两个符号**，**并补了一个我们留的开放题的实际测量**。

**① 数字更正（重建脚本、跑全 6 个组合，我们见其中 4 个）**

| 组合器 | 旧值（错） | 新值（对） |
|---|---|---|
| LightGBM stacker | +0.00002 | **−0.000059** |
| bagged hill climb | −0.00014 | **+0.000066** |
| weighted rank avg | +0.000063 | +0.000063（不变） |
| CatBoost stacker（新跑） | — | **−0.000057** |

核心结论没变：**没有任何组合器能稳定胜过加权 rank 平均**。但符号改对后图样更干净——四个组合器全挤在 −0.000059 ~ +0.000066 的窄带里、rank 平均 +0.000063 就在带中间。

**② 尺度那条他补了第二条理由**
我们的"别拿我的尺子量你的数据"= 他的数是 CV delta、我们的 5e-5 是 LB 估计，不是一个尺度。他回：**对，而且我自己的数也是 CV delta，跟你的 5e-5 也不是一个尺度，我不会把它们放一个尺度上**。→ 两条独立理由都指向"别合并尺度"，双方已完全合流。

**③ 他真的测了我们留的开放题（test 列同时是 K 模型平均）**
同一个构造，一次每个 test 行平均 5 个折模型、一次平均 15 个（3 组折堆叠）。公开分都是 **0.94638，5 位小数完全相同** → 本 episode 上 5→15 模型平均对 test 列的影响 **< 5e-6**（低于最后一位分辨率）。
他的限定：这**只**钉住了 test 这一半（OOF 侧在那对实验里没动），所以 OOF→test 迁移案仍没闭环；但**至少把"test 列在不同 K 下是不同对象"的担心压到了 < 5e-6**。

**回复草稿 v6（2026-09-17 —— 可直接整段粘贴，187 词 / 3 段）**

> AI 味自查（同套规则）：破折号 **0**、弯引号 **0**、弯撇号 **0**、`1/sqrt(2)` 无；有缩写（`isn't` / `wouldn't` / `I'd`）；段落长短不均；无连续金句收尾。

```
Thanks for rerunning it. The sign correction is the honest move, and it makes your conclusion cleaner: all four combiners now sit in a band from -0.000059 to +0.000066, with the weighted rank average at +0.000063 inside it rather than above it. So "no combiner beats the rank average" is really "everything is within noise of it and of zero" - and it survives the fix.

On the scale point we're aligned, and you went one step past me: my "no ruler for yours" was about your numbers against my ruler, but your own deltas aren't on my scale either. Both reasons point the same way, so I wouldn't put them on one.

The 5-vs-15 test is the part I'd keep. It bounds the board-side half of the open axis cleanly: averaging more models moves the public score by under 5e-6 here, so the test column isn't a meaningfully different object between K=5 and K=15 on this episode. It leaves the OOF half untouched - the half neither of us has - so the transfer case stays open. A fair place to leave it.
```

- **回帖策略**：① 把"符号改对反而让结论更硬"点出来（四个组合器全在 ±7e-5 窄带、rank 平均在带中间 → "没人胜过 rank 平均" = "大家都在噪声里彼此不分"）；② 尺度那条已合流，确认即可；③ 5-vs-15 是他这轮最有价值的贡献，明确它**钉住哪一半、没钉住哪一半**，收尾落在"诚实地留在这里"。
- **没补合规保险句**：v5 已含 `Nothing in that needs anyone's approach to answer.`（他这轮没再追问方案），v6 不必重复。
- ⚠️ 从上方代码块复制、别从聊天渲染层复制；发出后回读——看引号是否还是直的、句号后有没有丢空格。

## 4. 发布顺序清单

- [x] 新建 Kaggle Notebook → File → **Import Notebook** → 选 `EV_S6E9_postmortem.ipynb`
- [x] 右侧 **Add Input → Competition → Playground S6E9**（必须，否则不进竞赛 Code 页）
- [x] **标题**：`S6E9 postmortem: CV up, LB down (49 models)`（43 字符，计数器未变红）
- [x] **没有描述框**：钩子靠 ① 标题 ② notebook 第一个 markdown 单元格 ③ Discussion 正文
- [x] **Run All** → 5 张图都有输出（日志已核对）
- [x] **Save Version**；图表随版本保存（`nbconvert` 写出 5 个图目录）
- [x] 检查版本页：图片可见、无报错
- [ ] 侧边栏补 **Tags**（5 个）← 还没做
- [ ] **License: MIT**（当前是 Apache 2.0，与仓库不一致）← 还没做
- [ ] （可选）重新 Import 本地 notebook → Save Version 得 v2（带上钩子段 + 仓库链接）
- [ ] 再发 Discussion 帖（可选但门槛低）
- [ ] 之后**不要**在任何地方请求投票
