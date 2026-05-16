#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
    exit 1
  fi
fi

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

# 迭代04-B：3×HyFormer + 与 ns time test 一致的 dense/sparse/lr/dropout/wd；batch 192 减轻梯度噪声（OOM 则改为 176）
python3 -u "${CODE_ROOT}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --batch_size 192 \
    --num_hyformer_blocks 3 \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --dense_weight_decay 0 \
    --dropout_rate 0.01 \
    --patience 7 \
    --buffer_batches 20 \
    "$@"
