#!/bin/bash
# 生成提交评测容器用的 infer.py 独立副本（带版本注释）。不改仓库根目录 infer.py。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST_DIR="${ROOT}/hyper_log/dist"
mkdir -p "${DST_DIR}"
SRC="${ROOT}/infer.py"
OUT="${DST_DIR}/infer_eval_snapshot.py"

branch="$(git -C "${ROOT}" branch --show-current 2>/dev/null || echo unknown)"
sha="$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

{
  printf '%s\n' '# -*- coding: utf-8 -*-'
  printf '%s\n' "# Snapshot of infer.py for Taiji Model Evaluation upload."
  printf '%s\n' "# Generated: ${ts}"
  printf '%s\n' "# Repo branch: ${branch}"
  printf '%s\n' "# Repo commit: ${sha}"
  printf '%s\n' "# Source file: infer.py (repository root)"
  printf '%s\n' "#"
  printf '%s\n' "# MUST ship with compatible dataset.py / model.py / utils.py from the SAME commit."
  printf '%s\n' ""
  cat "${SRC}"
} > "${OUT}"

echo "Wrote ${OUT}"
ls -la "${OUT}"
