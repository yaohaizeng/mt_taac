#!/bin/bash
# 迭代03-A：正则加强版（2-block），对应 submit_bundle_iter03_reg
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

chmod +x "${ROOT}/hyper_log/bundle_C_iter03_reg/run.sh"

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_C_iter03_reg/run.sh" \
  --name taac_baseline_iter03_reg \
  --description '迭代03-A：iter02_v2后验证AUC峰~0.868后回落→加强wd0.02/drop0.06/lr8e-5/sparse0.036/patience12，2-block' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_iter03_reg"

echo "OK: ${ROOT}/hyper_log/submit_bundle_iter03_reg"
