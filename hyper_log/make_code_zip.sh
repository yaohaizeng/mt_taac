#!/bin/bash
# 在仓库根目录生成用于 Taiji 上传的 code.zip（排除 outputs、抓取缓存、大体积 zip 等）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/hyper_log/code_upload.zip}"
cd "${ROOT}"
rm -f "${OUT}"
zip -r "${OUT}" . \
  -x '*.git*' \
  -x 'outputs/*' \
  -x 'outputs/**' \
  -x 'Megatron-LM/*' \
  -x 'Megatron-LM/**' \
  -x 'hyper_log/*' \
  -x 'hyper_log/**' \
  -x '*.parquet' \
  -x '.venv/*' \
  -x 'venv/*' \
  -x '__pycache__/*' \
  -x '*.pyc' \
  -x '.cursor/*' \
  -x '.DS_Store'
echo "Wrote ${OUT}"
ls -la "${OUT}"
