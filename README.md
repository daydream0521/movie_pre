# MovieLens Explorer

> 课程题目：**推荐系统如何“猜你喜欢”：从协同过滤到深度学习推荐**
> 用 MovieLens 100K 的真实评分，讲清 **User-CF（找相似的人）** 与 **Item-CF（找相似的电影）**，并给出可现场演示、可逐步解释、指标来自实跑结果的 Streamlit 应用。

---

## 1. 一分钟跑起来

```bash
# ① 环境（本机已复用 conda base 环境，Python 3.12.7）
conda activate base
python -m pip install -r requirements.txt        # 依赖已装好可跳过

# ② 下载数据（约 5 MB，已下载可跳过；断网时也能跑）
python scripts/download_data.py
python scripts/download_data.py --check          # 确认 u.data / u.item 就绪

# ③ 跑实验（生成 results/*.csv 与 results/figures/*.png，约 10 秒）
python scripts/run_experiments.py

# ④ 启动演示应用
python -m streamlit run app.py
# 浏览器打开 http://localhost:8501
```

没有 conda 时的等价做法：

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
python -m pip install -r requirements.txt
```

**环境说明**：本机复用了已有的 conda `base` 环境（Python 3.12.7，numpy 1.26.4 / pandas 2.2.2 / scipy 1.13.1 / streamlit 1.37.1 均已就绪），因此没有另建 venv。实际执行的命令、完整输出、启动日志与测试结果都记录在 **`docs/run_log.txt`** 中，可逐行对照。

只想要某个实验：

```bash
python scripts/run_experiments.py --only e1     # 五折算法比较
python scripts/run_experiments.py --only e2     # 邻居数 K
python scripts/run_experiments.py --only e3     # 训练记录减少
```

跑测试：

```bash
python -m pytest tests -q                       # 10 项关键正确性测试
python tests/test_app_smoke.py                  # 无头跑通 5 个页面
```

---

## 2. 目录结构

```text
movie_pre/
├── app.py                     # Streamlit 入口（5 个页面）
├── requirements.txt
├── .streamlit/config.toml     # 深色主题
├── src/
│   ├── data.py                # 读取 u.data/u.item、官方五折、训练矩阵
│   ├── collaborative_filter.py# User-CF / Item-CF / 向量化预测 / 推荐解释
│   ├── evaluation.py          # MAE·RMSE·覆盖率、E1/E2/E3 流程、作图
│   ├── tutorial.py            # 5×5 可手算小矩阵与步骤轨迹
│   └── ui.py                  # 配色、卡片、矩阵表格等渲染片段
├── scripts/
│   ├── download_data.py       # 下载/断点续传/解压/就绪检查
│   ├── run_experiments.py     # E1/E2/E3 批量实验
│   ├── shot.py                # 页面截图（人工检查视觉用，非必需）
│   └── shot_lab_steps.py      # 逐步截图（同上）
├── results/                   # 实跑输出的 CSV 与 PNG（已入库，供报告/PPT 引用）
├── tests/                     # 关键失败风险对应的小测试
├── docs/
│   ├── experiment_report.md   # 实验设置、实测数据、结论、局限
│   ├── presentation_outline.md# 12 页课堂展示结构
│   ├── speaker_notes.md       # 逐页讲稿、Demo 流程、12 个问答
│   ├── run_log.txt            # 本次真实执行日志（实验 + 启动 + 测试）
│   └── screenshots/           # 五个页面的实际截图（PPT 可直接用）
└── data/                      # MovieLens 原始数据（.gitignore，不入库）
```

> 与计划书建议结构相比只做了一处调整：项目直接放在工作区根目录，因为本目录本身就是为该项目创建的空目录，多套一层 `movielens-explorer/` 没有收益。额外增加的 `src/ui.py` 与 `scripts/shot*.py` 只服务于界面与视觉检查，不含算法逻辑。

---

## 3. 数据与划分（不掺水的地方）

| 项目 | 值 |
|---|---|
| 评分条数 | 100,000（1–5 分整数） |
| 用户 / 电影 | 943 / 1,682 |
| 评分均值 | 3.5299（4 分最多：34,174 条） |
| 矩阵密度 | 6.30%（u1.base 训练集 5.04%） |
| 每人评分中位数 | 65 |
| 每片评分中位数 | 27 |

* 课堂演示与详细分析统一使用 **u1.base（80,000 条）** 训练，`u1.test`（20,000 条）**只用于评估**。
* 正式指标在 **u1–u5 五折**上跑同一流程，报告均值 ± 标准差。
* 程序里训练矩阵由 `uN.base` 构建，测试评分从未进入相似度或均值计算（`tests/test_core.py` 有对应断言）。
* 原始数据不入 Git；`.gitignore` 已排除 `data/`。

---

## 4. 算法定义（代码、界面、报告、讲稿共用这一套）

设训练评分矩阵为 `R`，观测掩码 `M`（`M[u,i]=1` 表示评过分），`mean(u)` 只用该用户在训练集中**已观测**的评分。

**相似度**

- 用户向量：`x_u = (r(u,i) - mean(u))`，仅观测位置非零 → 余弦 = 中心化余弦。
- 物品向量：同样减去**该条评分所属用户**的均值 → 余弦 = 调整余弦（去掉了不同用户打分尺度的影响）。
- 收缩：`sim ← sim * n/(n+10)`，`n` 为两用户/两物品的共同评分数；`n < 3` 时置 0。对角线置 0。
- 只使用训练集评分计算；不使用类型标签或文本。

**预测（K 是预测时实际参与的正相似邻居上限）**

```
User-CF: pred(u,i) = mean(u) + Σ_{v∈N_K} sim(u,v)·(r(v,i)-mean(v)) / Σ|sim(u,v)|
Item-CF: pred(u,i) = mean(u) + Σ_{j∈N_K} sim(i,j)·(r(u,j)-mean(u)) / Σ|sim(i,j)|
```

- `N_K` 只含**相似度为正**、且能提供评分证据的邻居（User-CF：给 i 评过分的人；Item-CF：u 评过的电影），按相似度降序取前 K 个。
- 无可用邻居 → 回退 `mean(u)`；用户训练集为空 → 回退全局均值。回退次数在实验里单独统计。
- 预测裁剪到 `[1,5]`；推荐结果排除该用户训练集已评分的电影。
- 默认 K=30，敏感性实验 K∈{5,10,20,30,50}。

---

## 5. 实测结果（全部来自 `results/*.csv`）

E1，五个官方 fold，K=30：

| 模型 | MAE（均值±标准差） | RMSE（均值±标准差） | 覆盖率 | 回退率 |
|---|---|---|---|---|
| 用户均值基准 | 0.8362 ± 0.0090 | 1.0437 ± 0.0119 | 100% | 0% |
| **User-CF** | **0.7295 ± 0.0048** | **0.9370 ± 0.0061** | 99.71% | 0.29% |
| **Item-CF** | 0.7332 ± 0.0089 | 0.9415 ± 0.0112 | 99.19% | 0.82% |

* 两种 CF 都比均值基准好约 0.10 RMSE；User-CF 在 5 折中的 4 折优于 Item-CF，但差距（0.0045）小于折间波动（标准差 0.006–0.011），**不能据此断言谁绝对更强**。
* 构建相似度：用户约 0.03 s / 物品约 0.05 s；2 万条测试评分批量预测 0.12–0.18 s（纯 CPU、单进程；耗时随机器浮动，指标不受影响）。

E2（固定 u1，K 敏感性）：RMSE 随 K 从 5→20 明显下降（User-CF 0.9867→0.9477），K=20~30 达到最好，K=50 略回升（0.9480）。覆盖率基本不变（99.69% / 99.18%），因为回退只发生在“一个邻居都没有”时。

E3（固定 u1，训练记录保留 100%/50%/25%）：保留 25% 时 User-CF RMSE 0.9472→1.1145、覆盖率 99.69%→78.56%；Item-CF RMSE 0.9604→1.1804、覆盖率 99.18%→76.02%。训练数据变稀时，两种算法同时变差，覆盖率下降尤其剧烈。

> 细节、逐折数据与解读见 `docs/experiment_report.md`；结论表述严格对应上表，不写“某算法一定更好”。

---

## 6. 五个页面

| 页面 | 内容 |
|---|---|
| **概览** | 数据规模、评分分布、两种算法在问什么问题 |
| **原理实验室** | 5×5 小矩阵 + 5 步轨迹：矩阵 → 相似度 → Top-K → 加权 → 结论；可切算法、换目标、调 K，附手算复核对白 |
| **电影推荐** | 选真实用户 → 历史评分/评分分布/类型偏好 → Top-10（含邻居支撑数、回退标注）→ 单条推荐的加权依据 |
| **实验结果** | E1/E2/E3 的表格与图表，全部读 `results/*.csv`，附“怎么读”与原始文件下载 |
| **从 CF 到深度学习** | 相似度 → 矩阵分解 → Embedding+MLP/NCF 的概念过渡，明确本实验主线是传统 CF |

界面为深色投屏风格，**不使用任何海报图片**（不依赖联网），电影以标题 + 年份 + 类型标签呈现；配色、字体（Inter + 微软雅黑回退）统一在 `src/ui.py`。

演示时可直接用 URL 直达某页：`http://localhost:8501/?page=lab`（`overview` / `lab` / `recommend` / `results` / `concept`）。

---

## 7. 正确性自查（对应计划书验收标准）

| 验收点 | 如何保证 |
|---|---|
| 推荐不含已看电影 | `recommend()` 的候选集 = `~M[u]`；`test_recommend_excludes_seen_items` 断言 |
| 预测只用训练集 | 模型只由 `uN.base` 构建；`test_test_ratings_never_in_training_matrix` 断言 test∩train=∅ |
| 空评分不是 0 分 | 均值/相似度只对 `M` 生效；`test_missing_is_not_zero` |
| 无邻居要标注回退 | `PredictionResult.is_fallback` + 卡片“用户均值回退”徽标；`test_no_neighbor_falls_back_to_user_mean` |
| 公式与界面一致 | 手算页与 `CFModel.explain()` 同源；`test_user_cf_prediction_matches_formula` 等 3 项 |
| 结果是真跑的 | E1/E2/E3 写 CSV，页面只读 CSV；报告数字逐条对上 |
| 可手算的例子 | 5×5 矩阵：User-CF 预测 4.329、Item-CF 预测 4.368，推导见实验室“手算复核”或 `docs/experiment_report.md` §5 |

---

## 8. 已知限制

1. 评分预测误差低 ≠ 推荐体验好。E1 的 MAE/RMSE 衡量“猜分数”，Top-N 排序质量没有做正式离线评估（E4 可选，未实现）。
2. 相似度来自共现行为，冷启动用户/新电影没有邻居，只能回退均值。
3. 预测裁剪到 [1,5] 会让“少数强邻居都打 5 分”的条目触顶（界面会说明有多少条触顶）。
4. 训练记录减少的实验是**模拟实验**，用户集合不变，不等价于真实冷启动。
5. 计时为单机 CPU、单进程结果，仅用于量级讨论，不代表工程性能。
