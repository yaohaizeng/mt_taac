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

# 迭代03-A：在 iter02_fix_v2 基础上抑制验证 AUC 中后期回落（抓取峰值~0.868 后跌至~0.864）
python3 -u "${CODE_ROOT}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --num_hyformer_blocks 2 \
    --patience 12 \
    --dropout_rate 0.06 \
    --lr 8e-5 \
    --sparse_lr 0.036 \
    --dense_weight_decay 0.02 \
    --buffer_batches 32 \
    "$@"
