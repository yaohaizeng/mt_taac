#!/bin/bash

# 将脚本自身所在目录解析为绝对路径，保证从任意工作目录调用都能正确定位 train.py。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 将项目根目录加入 Python 模块搜索路径，使 train.py 能直接 import 同级模块
# （model, dataset, trainer, utils 等）而无需安装为 package。
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- P0 优化配置（2026-05-20）----
#
# 1) RankMixer full 模式：num_queries=1 + user_ns_tokens=5 + user_pair_hash_cross
#    → num_ns=12, T=1×4+12=16, 64%16=0 ✓
#    （对比 d_model=128 见 scripts/compare_p0_rankmixer.py；默认选 num_queries=1+d64，算力更省）
# 2) seq 超高基数特征：seq_use_hash_emb 替代零向量 skip
# 3) seq_d 专用 longer encoder（top_k=80），其余域保持 transformer
#
# 可选：在平台上运行 A/B 对比后切换 d_model
#   python3 scripts/compare_p0_rankmixer.py
#
# "$@" 将调用 run.sh 时附加的所有额外参数透传给 train.py。
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 1 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_user_pair \
    --user_pair_fids '62,63,64,65,66' \
    --user_pair_emb_dim 32 \
    --user_pair_use_hash_cross \
    --rank_mixer_mode full \
    --seq_encoder_types 'seq_a:transformer,seq_b:transformer,seq_c:transformer,seq_d:longer' \
    --seq_top_k 80 \
    --seq_use_hash_emb \
    --seq_hash_bucket_size 200000 \
    "$@"

# ---- 备选配置：GroupNSTokenizer，由 ns_groups.json 驱动 ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
