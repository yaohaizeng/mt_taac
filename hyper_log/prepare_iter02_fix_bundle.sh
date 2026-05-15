#!/bin/bash
# 仅生成「迭代 02 修复」提交目录：解压 code.zip + 与当前仓库一致的 train 参数（解决模板旧 train.py）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ZIP="${ROOT}/hyper_log/code_iteration04.zip"
CFG="${ROOT}/hyper_log/config.yaml"
URL='https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a'

[[ -f "$ZIP" ]] || { echo "缺少 $ZIP，请先: bash hyper_log/make_code_zip.sh $ZIP"; exit 1; }

cp "${ROOT}/run.sh" "${ROOT}/hyper_log/bundle_A_bce/run.sh"

taac2026 prepare-submit \
  --template-job-url "$URL" \
  --zip "$ZIP" \
  --config "$CFG" \
  --run-sh "${ROOT}/hyper_log/bundle_A_bce/run.sh" \
  --name taac_baseline_iter02_fix_v2 \
  --description '迭代02修复v2：解压code.zip + PYTHONPATH未设置时兼容(set -u)，避免 run.sh 启动即退出' \
  --run \
  --allow-dirty \
  --out "${ROOT}/hyper_log/submit_bundle_iter02_fix"

echo "OK: ${ROOT}/hyper_log/submit_bundle_iter02_fix"
