#!/bin/bash
set -euo pipefail

# ── Evaluation / 实验对齐用 baseline：复刻 Taiji 作业「ns time test」(91994) 的 train CLI。
# 特征：RankMixer NS + use_time_features（dense 时间）+ use_time_ns（离散时间 NS token）。
# 其余超参走 train.py 默认值（如 lr=1e-4、dropout=0.01、patience=5、dense_weight_decay=0、buffer_batches=20）。
#
# Taiji 上前身为直接跑 SCRIPT_DIR/train.py；现与 code.zip 解压目录对齐，避免模板旧代码。
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

python3 -u "${CODE_ROOT}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
