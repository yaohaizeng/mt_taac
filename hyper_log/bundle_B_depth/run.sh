#!/bin/bash
set -euo pipefail

# 将脚本自身所在目录解析为绝对路径，保证从任意工作目录调用都能正确定位 train.py。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Taiji：解压 code.zip，避免运行模板旧 train.py（版本不一致会直接 argparse 失败）。
CODE_ROOT="${SCRIPT_DIR}"
if [[ -f "${SCRIPT_DIR}/code.zip" ]]; then
  EX="${SCRIPT_DIR}/._taiji_code_extract"
  rm -rf "${EX}"
  mkdir -p "${EX}"
  if command -v unzip >/dev/null 2>&1; then
    unzip -qo "${SCRIPT_DIR}/code.zip" -d "${EX}"
  else
    python3 -c 'import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
      "${SCRIPT_DIR}/code.zip" "${EX}"
  fi
  if [[ -f "${EX}/train.py" ]]; then
    CODE_ROOT="${EX}"
  else
    sub="$(find "${EX}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
    if [[ -n "${sub}" && -f "${sub}/train.py" ]]; then
      CODE_ROOT="${sub}"
    fi
  fi
  if [[ ! -f "${CODE_ROOT}/train.py" ]]; then
    echo "[run.sh] ERROR: 解压 code.zip 后未找到 train.py。" >&2
    ls -la "${EX}" >&2 || true
    exit 1
  fi
fi

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

# ---- 并行方向 B：模型容量（加深 HyFormer Block），非损失函数改动 ----
# 与任务 A 区分：num_hyformer_blocks 3；略降 dense/sparse lr 以稳住更深网络。
# batch_size 必须小于默认 256：多一层 HyFormer + 平台 GPU 常与其它进程共享，256 会 CUDA OOM（见作业 106475 日志）。
python3 -u "${CODE_ROOT}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --batch_size 160 \
    --num_hyformer_blocks 3 \
    --patience 10 \
    --dropout_rate 0.045 \
    --lr 8e-5 \
    --sparse_lr 0.038 \
    --dense_weight_decay 0.01 \
    --buffer_batches 32 \
    "$@"
