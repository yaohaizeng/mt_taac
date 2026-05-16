#!/bin/bash
# 迭代05：评测对齐 baseline（ns time test）显式超参，单次提交目录 submit_bundle_iter05_ns_strict
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

chmod +x "${ROOT}/hyper_log/bundle_G_iter05_ns_strict/run.sh"

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_G_iter05_ns_strict/run.sh" \
  --name taac_baseline_iter05_ns_strict \
  --description '迭代05：显式对齐ns time默认(dropout0.01/patience5/lr1e-4等)，撤销iter04_mild微调假设' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_iter05_ns_strict"

echo "OK: ${ROOT}/hyper_log/submit_bundle_iter05_ns_strict"
