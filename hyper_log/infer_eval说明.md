# 迭代 04-A / 04-B 与 `infer.py` 推理说明

## 能否直接用仓库里的 `infer.py`？

**可以。** 评测容器里使用同一套  
`/Users/aaa/Desktop/taac_baseline/infer.py`（当前分支 **`tuning_hyperpara`** 上的版本即可），**无需**为 04-A、04-B 各写一份推理脚本。

原因简述：

- `infer.py` 从 **`MODEL_OUTPUT_PATH`** 指向的 checkpoint 目录读取 **`train_config.json`**（训练保存 best 时由 `trainer.py` 写入），据此还原 **`num_hyformer_blocks`**、NS tokenizer、`use_time_features` / `use_time_ns` 等结构参数。
- **04-A**（2-block）与 **04-B**（3-block）的差别已写进各自 ckpt 的 `train_config.json`，同一份 `infer.py` 会对权重做 **strict** `load_state_dict`。

## 必须满足的前提

1. **`MODEL_OUTPUT_PATH`** 指向 **`best_model` 目录**（内含 `model.pt`、`train_config.json`、`schema.json` 等 sidecar）。
2. **评测环境中的 `dataset.py`、`model.py`、`utils.py`（及 `infer.py` 依赖的其它模块）与训练打包进 `code.zip` 的版本一致**（同一 commit / 同一分支快照）。否则可能出现 API 或 shape 不匹配。
3. 环境变量：`EVAL_DATA_PATH`、`EVAL_RESULT_PATH` 按平台要求设置。

## 单独生成用于上传的 `infer.py` 副本

仓库根目录 **`infer.py`** 保留为唯一维护源；提交评测或归档时生成快照：

```bash
cd /Users/aaa/Desktop/taac_baseline
bash hyper_log/snapshot_infer_for_eval.sh
```

输出：**`hyper_log/dist/infer_eval_snapshot.py`**（文件头含生成时间、branch、commit）。

> 快照 **不能**单独替代 `dataset.py` / `model.py`：评测镜像需与训练代码同步更新。

## 后续若 `infer.py` 有改动

已添加 Cursor 规则：编辑 **`infer.py`** 时助手应 **提醒你**更新评测侧代码，并建议执行 **`hyper_log/snapshot_infer_for_eval.sh`** 重新生成 `infer_eval_snapshot.py`。
