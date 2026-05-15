#!/bin/bash
set -euo pipefail

# 将脚本自身所在目录解析为绝对路径，保证从任意工作目录调用都能正确定位 train.py。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Taiji：runtime 目录里会有模板下发的旧版 train.py/trainer.py，同时也有你上传的 code.zip。
# 必须解压 code.zip 并以其中源码为准执行 train.py，否则会报 unrecognized arguments 等版本不一致错误。
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
    echo "[run.sh] ERROR: 解压 code.zip 后未找到 train.py，请检查 zip 是否从仓库根目录打包。" >&2
    ls -la "${EX}" >&2 || true
    exit 1
  fi
fi

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

# ---- RankMixer NS（T=16, d_model=64）----
# 迭代 02：抗过拟合（BCE + AdamW weight_decay）
python3 -u "${CODE_ROOT}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --patience 10 \
    --dropout_rate 0.05 \
    --lr 9e-5 \
    --sparse_lr 0.04 \
    --dense_weight_decay 0.01 \
    --buffer_batches 32 \
    "$@"
