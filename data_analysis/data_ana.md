# 真实训练集数据分析汇总

> 数据来源：平台任务日志 `data_analysis/data.log`（TAIJI / AMS 2026，2026-05-17 运行）  
> 数据路径：`TRAIN_DATA_PATH=/data_ams/industrial_training_data`  
> 分析脚本：训练启动时 `train.py` 内建的 `[DATA]` schema / batch 统计输出

---

## 1. 数据规模与切分

| 指标 | 数值 |
| --- | --- |
| Parquet 文件数 | 1000 |
| 训练样本数 | **1,895,650** |
| 验证样本数 | **204,306** |
| 合计 | **2,099,956** |
| 切分方式 | Row Group 粒度：940 个 RG 训练 + 104 个 RG 验证（`valid_ratio=0.1`） |
| 训练 `batch_size` | 256 |
| 每 epoch 训练 step | 8030 |
| 首 batch 时间戳范围（UTC） | 2026-02-28 15:42 ~ 2026-03-04 01:31（约 **3.4 天**） |

验证集与训练集时间窗口在日志中重叠区间一致（验证首 batch：`1772294124` ~ `1772588787`），说明按 **Row Group 顺序尾部** 切分，而非严格按时间 hold-out；与 `dataset.py` 设计一致。

---

## 2. 标签与类别不平衡（PCVR）

任务为二分类：`label_type == 2` 为正例（转化），`dataset` 映射为 `label=1`。

| 集合 | 首 batch 正例率 | 首 batch 正例数 / batch_size |
| --- | --- | --- |
| 训练（`train_first_batch`） | **8.98%** | 23 / 256 |
| 验证（`valid_first_batch`） | **6.25%** | 16 / 256 |

说明：

- 全量精确 CVR 需在全数据上统计；日志仅给出 **首个 batch** 的瞬时比例。
- 验证首 batch 正例率低于训练首 batch，与尾部 RG 分布略偏负例一致。
- 当前训练配置为 **`loss_type=bce`**（默认）；正负约 1:10~1:16 量级时，可后续尝试 `--loss_type focal`。

---

## 3. 特征 Schema 总览

Schema 文件：`/data_ams/industrial_training_data/schema.json`（已 dump 至任务 `log/schema_dump.json`）。

### 3.1 非序列特征

| 分组 | 特征数 | 展平维度 / 说明 | 词表规模（max_vocab） |
| --- | --- | --- | --- |
| `user_int` | 46 | flat_dim=**476** | max_vocab=2862 |
| `item_int` | 14 | flat_dim=**33** | max_vocab=25495 |
| `user_dense` | 10 | 原始；`split_user_dense=True` 后保留 **5** 个 fid，**total_dim=606** | 见下 |

`split_user_dense=True` 时：

- 保留进入 `user_dense_feats` 的 5 个 fid（日志未逐条列出，paired 部分单独处理）。
- **配对稠密特征** `user_pair_fids = [62, 63, 64, 65, 66]`，池化方式 `pair_pooling=relu_weighted`。
- 用户 UE 类 fid：`user_ue_fids = [61, 87]`（由 RankMixer NS / 其他模块使用，见训练 Args）。

NS Tokenizer（RankMixer）：

| 侧 | fid 数 | total_emb_dim | num_ns_tokens |
| --- | --- | --- | --- |
| User | 46 | 2944 | 4 |
| Item | 14 | 896 | 2 |

另：**时间 NS token** 1 个（`use_time_ns=True`），合计 **num_ns=8**。

### 3.2 序列域（4 条异构行为链）

日志中 `seq_*` 与 Parquet 列前缀 `domain_*_seq` 的对应关系：

| 模型域 | Parquet 前缀 | 时间戳 fid | side 特征数 | 截断长度（训练配置） |
| --- | --- | --- | --- | --- |
| seq_a | `domain_a_seq` | 39 | 9 | 256 |
| seq_b | `domain_b_seq` | 67 | 14 | 256 |
| seq_c | `domain_c_seq` | 27 | 12 | 512 |
| seq_d | `domain_d_seq` | 26 | 10 | 512 |

各域 **fid → vocab_size**（schema 声明，0 表示时间戳列无 embedding 词表）：

**seq_a（domain_a，fid 38–46）**

| fid | vocab | 备注 |
| --- | --- | --- |
| 38 | 889,551 | 高基数 ID 类 |
| 39 | 0 | 时间戳 |
| 40 | 19 | 低基数（行为类型类） |
| 41 | 12 | |
| 42 | 1,028 | |
| 43 | 3,498 | |
| 44 | 13,692 | |
| 45 | 8,175 | |
| 46 | 18 | |

**seq_b（domain_b，fid 67–79, 88）**

| fid | vocab | 备注 |
| --- | --- | --- |
| 67 | 0 | 时间戳 |
| 68 | 28 | 行为类型类（约 28 类） |
| 69 | **85,946,394** | **emb_skip**（>1M 阈值） |
| 70–79, 88 | 740 ~ 531,477 | 含多个十万~百万级 |

**seq_c（domain_c，fid 27–37, 47）**

| fid | vocab | 备注 |
| --- | --- | --- |
| 27 | 0 | 时间戳 |
| 28 | 75 | |
| 29 | 6,551,861 | **emb_skip** |
| 30 | 864 | |
| 31 | 6,896 | |
| 32 | 7 | 极低基数（类似 action_type） |
| 33 | 5 | |
| 34 | 1,292,419 | **emb_skip** |
| 35 | 2,998 | |
| 36 | 1,134,859 | **emb_skip** |
| 37 | 9,697 | |
| 47 | 125,610,776 | **emb_skip** |

**seq_d（domain_d，fid 17–26）**

| fid | vocab | 备注 |
| --- | --- | --- |
| 17 | 5 | 极低基数（与 demo 中 action_type 1–4 同类） |
| 18 | 980 | |
| 19 | 3,469 | |
| 20 | 11,510 | |
| 21 | 5,156 | |
| 22 | 443,147 | |
| 23 | 630,399 | |
| 24 | 626 | |
| 25 | 15 | |
| 26 | 0 | 时间戳 |

**Embedding 跳过（`emb_skip_threshold=1_000_000`）**

- seq_b：13 个 side 特征中 **跳过 1 个**（fid 69）
- seq_c：11 个 side 特征中 **跳过 4 个**（fid 29, 34, 36, 47）

跳过后前向以 **零向量** 代替，节省显存；序列长度与截断仍正常参与 Attention。

---

## 4. 序列长度分布（首 batch 统计）

配置截断：`seq_a/b:256`，`seq_c/d:512`。下表为日志中 **训练首 batch** 与 **验证首 batch** 的 `*_len` 统计。

| 域 | 训练 mean | 训练 max | 训练 min | 验证 mean | 验证 max |
| --- | --- | --- | --- | --- | --- |
| seq_a | 220.8 | 256 | 13 | 216.7 | 256 |
| seq_b | 201.1 | 256 | **0** | 199.4 | 256 |
| seq_c | 353.4 | 512 | 23 | 352.4 | 512 |
| seq_d | 485.8 | 512 | **0** | 491.2 | 512 |

要点：

- **seq_d** 平均长度最接近上限 512，历史行为最长，计算最重。
- **seq_b / seq_d** 存在 `min=0` 的空序列样本，依赖 padding mask；MeanPool 与 Attention 已按 mask 处理。
- 均值均显著低于 max，截断主要影响长尾用户；seq_d 约 95% 利用率（485/512）。

---

## 5. 稀疏度与其他 batch 统计

| 字段 | shape | 稀疏/分布 |
| --- | --- | --- |
| `user_int_feats` | [256, 476] | zero_frac ≈ **84.2%**（训练）/ 84.8%（验证） |
| `item_int_feats` | [256, 33] | zero_frac ≈ **69.2%** |
| `user_dense_feats` | [256, 606] | mean ≈ -0.0025, std ≈ 0.107（已归一化量级） |
| `user_int` max_id | — | 2849（< max_vocab 2862） |
| `item_int` max_id | — | 22859（< max_vocab 25495） |

首 batch **无 OOB**（out-of-bound）特征 ID，说明 schema 词表与数据一致。

---

## 6. 与 demo（1000 条）对比

| 维度 | HF demo_1000 | 平台真实训练集（本日志） |
| --- | --- | --- |
| 样本量 | 1,000 | **~210 万** |
| 列布局 | 120 列 flat（domain_*_seq） | 相同 schema 体系（prefix 一致） |
| 首 batch CVR | label_type=2 约 **12.4%** | 首 batch 约 **9.0%**（训练） |
| seq_d 平均长度 | ~1185（未截断统计） | mean **~486**（截断 512 后） |
| 高基数 seq 特征 | demo 未 skip | **5 个 fid** 因 >1M 跳过 Embedding |

真实数据序列更短（截断 + 分布不同），正例率与 demo 不完全可比；**调参应以本日志对应的全量统计为准**。

---

## 7. 本次运行的训练配置摘要

与数据分析相关的关键 Args（摘自日志）：

| 项 | 值 |
| --- | --- |
| `seq_encoder_type` | transformer |
| `num_queries` | 2 |
| `num_hyformer_blocks` | 2 |
| `d_model` / `emb_dim` | 64 |
| `T`（RankMixer token 数） | **16**（2×4 序列 + 8 NS） |
| `loss_type` | bce |
| `batch_size` | 256 |
| `num_workers` | 8 |
| `buffer_batches` | 20 |
| `use_amp` | True（bf16） |
| `emb_skip_threshold` | 1,000,000 |

模型参数量：**198,664,961**（Sparse Embedding ~196.2M，Dense ~2.48M）。

---

## 8. 训练初期效果（日志前 4 epoch）

| Epoch | avg_train_loss | Valid AUC | Valid LogLoss |
| --- | --- | --- | --- |
| 1 | 0.2230 | 0.8648 | 0.2282 |
| 2 | 0.2171 | 0.8684 | 0.2261 |
| 3 | 0.2150 | 0.8700 | 0.2250 |
| 4 | 0.2134 | **0.8702** | **0.2243** |

每个 epoch 结束后 **重初始化 96 个高基数 Embedding**（`reinit_sparse_after_epoch=1`），仅保留 1 个低基数表状态。

---

## 9. 建模与调参建议（基于本日志）

1. **类别不平衡**：首 batch CVR ~9%，默认 BCE 可跑通；若正类召回不足，可试 `focal` + 调整 `focal_alpha`。
2. **序列算力**：优先关注 **seq_d**（最长、接近 512）；可试 `longer` + `seq_top_k=50` 降 FLOPs，或保持 `transformer` 换精度。
3. **空序列**：seq_b/seq_d 存在长度为 0 的样本，属正常；无需改数据，注意 MeanPool 分母 `clamp(min=1)`。
4. **高基数特征**：5 个 seq 侧 mega-vocab 已 skip，不要强行减小 `emb_skip_threshold` 除非显存充足。
5. **验证切分**：当前为 RG 尾部 10%，非纯时间切分；若要做时间外推验证，需改 `dataset` 切分策略或单独 hold-out 日期。
6. **seq_d fid=17（vocab=5）**：极低基数，更像行为类型而非 item id；与此前 demo EDA「多行为类型序列」结论一致，四条域均为 **多行为历史**，非单一转化链。

---

## 10. 日志中的非数据项（可忽略）

- TAIJI 容器启动、`cp`/`touch` 失败、`libvgpu` debug、TensorFlow oneDNN 警告等与数据内容无关。
- `initCmd failed` 后任务仍正常进入 `run.sh` 训练。

---

*文档生成说明：指标均直接摘录或由 `data.log` 中 `[DATA]` 行与 Args 行推导；若需全量 CVR、序列长度分位数，建议在平台上对 `PCVRParquetDataset` 增加离线扫描脚本复用同一 schema。*
