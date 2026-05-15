#!/bin/bash
# 生成「另一条优化方向」单独提交目录（加深 HyFormer，与迭代02修复假设不同）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_B_depth/run.sh" \
  --name taac_baseline_alt_depth3_v2 \
  --description '对照深度v2：3xHyFormer + batch_size=160（修复106475 CUDA OOM），lr/sparse/dropout同向微调，解压code.zip' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_alt_depth"

echo "OK: ${ROOT}/hyper_log/submit_bundle_alt_depth"
