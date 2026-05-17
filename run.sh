#!/bin/bash

# 将脚本自身所在目录解析为绝对路径，保证从任意工作目录调用都能正确定位 train.py。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 将项目根目录加入 Python 模块搜索路径，使 train.py 能直接 import 同级模块
# （model, dataset, trainer, utils 等）而无需安装为 package。
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- 当前激活配置：RankMixer NS Tokenizer（无需 ns_groups.json）----
# 用 RankMixer 模式将 NS（Non-Sequential）特征切分为等长 token：
#   - 将所有 User 侧整型 Embedding 拼接后均匀切成 4 个 token（--user_ns_tokens 4）
#   - 将所有 Item 侧整型 Embedding 拼接后均匀切成 2 个 token（--item_ns_tokens 2）
#   - 额外启用 1 个离散时间 NS token（由 hour/dow 离散 Embedding 生成）
# 该模式不依赖 ns_groups.json，适合快速启动或无分组配置文件的场景。
#
# 关键超参说明：
#   --num_queries 2        每条行为序列独立生成 2 个 Global Query Token
#                          Token 总数 T = num_queries * num_seq_domains + num_ns
#                          T 需满足 d_model % T == 0（RankMixerBlock token mixing 约束）
#   --emb_skip_threshold 1000000
#                          vocab_size > 100万 的超高基数特征跳过 Embedding 分配，
#                          前向时以零向量代替，节省 GPU 显存
#   --num_workers 8        DataLoader 并行读取 Parquet 数据的进程数
#   --use_amp --use_compile  bf16 混合精度 + torch.compile（默认已开启，可用
#                          --no-use_amp / --no-use_compile 关闭）
#   split_user_dense       UE(61,87)+时间走 user_dense_proj；62-66 的 dense 作
#                          user_int_weights（ReLU 加权 mean）；89-91 留在 dense
#   "$@"                   将调用 run.sh 时附加的所有额外参数透传给 train.py，
#                          例如：bash run.sh --batch_size 512 --lr 3e-4
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --split_user_dense \
    --user_ue_fids '61,87' \
    --user_pair_fids '62,63,64,65,66' \
    --use_amp \
    --use_compile \
    "$@"

# ---- 备选配置：GroupNSTokenizer，由 ns_groups.json 驱动 ----
# 使用 ns_groups.json 中的语义分组方案（7 个 User 组 + 4 个 Item 组），
# 每组特征通过一个独立的 FFN 投影为 1 个 NS Token，共生成 12 个 NS Token。
#
# 约束推导（d_model=64 时）：
#   num_ns = 7(user_int 组) + 1(user_dense) + 4(item_int 组) = 12
#   T = num_queries * num_seq_domains + num_ns
#   = num_queries * 4 + 12
#   需满足 d_model % T == 0，即 64 % T == 0
#   → T ∈ {1,2,4,8,16,32,64}，其中 T=16 时 num_queries=1 ✓（4*1+12=16）
#   → num_queries=2 → T=20，64%20≠0，不满足约束 ✗
#   故此模式下 num_queries 只能取 1。
#
# 切换方式：注释掉上方 RankMixer 块，取消注释下方 group 块。
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
