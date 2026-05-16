#!/bin/bash
# 迭代04-A：向 baseline 优化对齐 + 轻度 dropout（submit_bundle_iter04_mild）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

chmod +x "${ROOT}/hyper_log/bundle_E_iter04_mild/run.sh"

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_E_iter04_mild/run.sh" \
  --name taac_baseline_iter04_mild \
  --description '迭代04-A：对齐ns time lr/sparse/wd，dropout0.02轻正则；修正iter03过重正则低估评测AUC' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_iter04_mild"

echo "OK: ${ROOT}/hyper_log/submit_bundle_iter04_mild"
