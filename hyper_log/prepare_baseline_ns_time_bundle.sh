#!/bin/bash
# 生成与 Taiji「ns time test」(91994) 对齐的提交目录（baseline 复现实验用）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

chmod +x "${ROOT}/hyper_log/bundle_baseline_ns_time/run.sh"

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_baseline_ns_time/run.sh" \
  --name taac_baseline_ns_time_repro \
  --description 'Baseline复现：对齐ns time test(91994) RankMixer+time_ns，其余train.py默认' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_baseline_ns_time"

echo "OK: ${ROOT}/hyper_log/submit_bundle_baseline_ns_time"
