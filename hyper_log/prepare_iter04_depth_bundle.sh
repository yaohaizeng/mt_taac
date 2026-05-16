#!/bin/bash
# 迭代04-B：3-block + baseline式优化器 + batch192（submit_bundle_iter04_depth）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

chmod +x "${ROOT}/hyper_log/bundle_F_iter04_depth/run.sh"

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_F_iter04_depth/run.sh" \
  --name taac_baseline_iter04_depth \
  --description '迭代04-B：3xHyFormer batch192（OOM改176）；lr1e-4/sparse0.05/drop0.01/wd0 对齐ns time，修正iter03深度+强正则' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_iter04_depth"

echo "OK: ${ROOT}/hyper_log/submit_bundle_iter04_depth"
