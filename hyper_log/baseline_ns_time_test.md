# Baseline：Taiji「ns time test」训练作业

当前约定：**线下 TensorBoard 与其它训练任务对比时，以及线上 Model Evaluation 最优参照**，均以 **`ns time test`** 为准（你已确认评测效果最好）。

## Taiji 标识

| 字段 | 值 |
|------|-----|
| 作业名称 | `ns time test` |
| 说明 | `dense time feature + ns_time feature` |
| 内部 ID | **91994** |
| taskId | `angel_training_ams_2026_1029735554728158280_20260511235848_bdee12a5` |

## 抓取到的训练曲线摘要（验证集）

来源：`outputs/taiji-output/training/all-metrics-long.csv`（TensorBoard 同步）。

| 指标 | 代表性结果 |
|------|------------|
| **AUC/valid 峰值** | **≈ 0.86882**（step **38050** 附近） |
| **LogLoss/valid 低谷** | **≈ 0.22578**（同 step 一带） |
| 现象 | 峰值后验证 AUC 仍缓慢回落（与多数长跑一致），最佳 ckpt 宜取峰附近 |

## 最佳 Checkpoint（抓取列表）

`outputs/taiji-output/training/all-checkpoints.csv`：

- 目录名：`global_step38050.layer=2.head=4.hidden=64.best_model`
- 标记：`best=true`

## 训练入口（与线上抓取一致）

抓取文件：`outputs/taiji-output/training/code/angel_training_ams_2026_1029735554728158280_20260511235848_bdee12a5/files/run.sh`

要点：

- **RankMixer**：`--user_ns_tokens 4`、`--item_ns_tokens 2`、`--num_queries 2`、`--ns_groups_json ""`
- **`--emb_skip_threshold 1000000`**、`--num_workers 8`
- **未**在 `run.sh` 中指定 `lr` / `dropout` / `patience` / `dense_weight_decay` 等 → 使用当时 **`train.py` 默认值**（当前仓库默认：`lr=1e-4`、`dropout_rate=0.01`、`patience=5`、`dense_weight_decay=0`、`buffer_batches=20`、`sparse_lr=0.05`）
- **时间**：`train.py` 默认 **`use_time_features=True`**、`**use_time_ns=True**`（连续时间特征 + 离散时间 NS token），与作业说明一致

## 本地复现 / 新提交

使用 **`hyper_log/bundle_baseline_ns_time/run.sh`**：CLI 与上述 baseline 一致，并带 **`code.zip` 解压**（适配现行 Taiji）。

```bash
bash hyper_log/make_code_zip.sh hyper_log/code_iteration04.zip
bash hyper_log/prepare_baseline_ns_time_bundle.sh
# 再 taac2026 submit ... --bundle hyper_log/submit_bundle_baseline_ns_time ...
```

## 与其它线的关系

- **`taac_baseline_iter02_fix_v2`** 等：在 baseline **之上**做 BCE + `dense_weight_decay`、专用 lr/dropout 等消融；对比时应说明相对 **91994** 的差异。
- 后续迭代文档中的「对照」「提升」若无特别声明，均指相对 **本 baseline**。
