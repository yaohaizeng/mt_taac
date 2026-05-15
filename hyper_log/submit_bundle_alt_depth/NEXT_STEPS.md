# Taiji Submit Next Steps

This directory was prepared by `prepare-taiji-submit.mjs`.

## Intended live workflow

1. Open the template Job URL in a logged-in browser.
2. Copy the template Job.
3. Replace code zip, config file, `run.sh` with the files in `files/`.
4. Confirm the new `run.sh` entrypoint matches this experiment.
5. Fill Job Name and Job Description from `manifest.json`.
6. Submit the copied Job.
7. If `runAfterSubmit` is true, start the new Job and record the Job ID / instance ID.

## Prepared values

- Template Job URL: https://taiji.algo.qq.com/training/ckpt/angel_training_ams_2026_1029735554728158280_20260513203950_dfc2d95e/101210/95a8ce1c9e20cbf6019e2159d28f016a
- Job Name: taac_baseline_alt_depth3_v2
- Job Description: 对照深度v2：3xHyFormer + batch_size=160（修复106475 CUDA OOM），lr/sparse/dropout同向微调，解压code.zip
- Run after submit: true
- Code zip: files/code_iteration04.zip
- Config: files/config.yaml
- run.sh: files/run.sh

## Automation note

Live API/browser submission is intentionally not executed by this preparation tool.
Before enabling it, capture one successful manual Copy Job -> upload zip/config -> submit -> run flow from DevTools, including upload endpoints and request payloads.

