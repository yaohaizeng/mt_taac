"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """将 FeatureSchema 转换为模型所需的 feature_specs 列表。

    每条记录格式为 (vocab_size, offset, length)：
      - vocab_size：该特征在 flat token 向量中占据的所有位置里词表大小的最大值，
        用于构造对应的 nn.Embedding；
      - offset / length：该特征在 flat 输入向量中的起始位置和长度，
        供模型 forward 时切片取值。

    取 max 而非逐位建表的原因：同一逻辑特征可能被编码为多个连续 int 位（如多值特征），
    共用同一张 Embedding 表时以最大 vocab_size 为准，确保索引不越界。
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        # 取该特征占据的所有 token 位中最大的 vocab_size，保证 Embedding 不越界
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    """解析命令行参数，环境变量优先于 CLI 标志。

    参数按功能分为六组：
      1. 路径（data_dir / ckpt_dir / log_dir）
      2. 训练超参（batch_size / lr / num_epochs / patience / seed / device）
      3. 数据管道（num_workers / buffer_batches / train_ratio / valid_ratio / seq_max_lens）
      4. 模型结构（d_model / emb_dim / num_queries / num_hyformer_blocks / …）
      5. 损失函数（loss_type / focal_alpha / focal_gamma）
      6. 稀疏优化器与 Embedding 构造控制（sparse_lr / emb_skip_threshold / …）
    """
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # ── 1. 路径 ──────────────────────────────────────────────────────────────
    # 三条路径均可由同名环境变量覆盖（见函数末尾），方便容器/平台统一注入。
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # ── 2. 训练超参 ───────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    # num_epochs 设为 999 是一个"实质无上限"的哨兵值，实际由 early stopping 控制终止。
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # ── 3. 数据管道 ───────────────────────────────────────────────────────────
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    # buffer_batches 控制流式 shuffle 缓冲区大小（单位：batch 数），
    # 越大随机性越好但内存占用越高。
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    # valid_ratio 从所有 Row Group 的尾部切分，与训练集不重叠。
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    # eval_every_n_steps=0 表示仅在 epoch 结束时评估，>0 则额外在步内周期评估。
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    # seq_max_lens 按域独立截断，避免长尾序列撑大 padding 开销。
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # ── 4. 模型结构 ───────────────────────────────────────────────────────────
    # d_model 是 Transformer 主干的隐层维度，也是 RankMixerBlock token mixing 的约束基准：
    #   要求 d_model % T == 0，其中 T = num_queries * num_seq_domains + num_ns。
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    # num_queries：每个行为序列域独立生成的 Global Query Token 数量。
    # 与序列域数及 NS token 数共同决定 T，需满足 d_model % T == 0。
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--query_pooling', type=str, default='mean',
                        choices=['mean', 'din'],
                        help='Query generation pooling over each behavior sequence: '
                             'mean = masked mean pool, din = target-item cross-attention')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    # seq_encoder_type 决定序列侧编码器变体：
    #   swiglu     — 纯 FFN，无注意力，速度最快；
    #   transformer — 标准多头自注意力；
    #   longer     — Top-K 压缩编码器，仅此变体使用 seq_top_k / seq_causal。
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    # action_num=1 对应单任务二分类（PCVR），>1 时输出多标签 logit 向量。
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    # use_time_buckets：为序列中每个行为附加时间桶 Embedding，捕捉行为时效性。
    # 桶边界由 dataset.BUCKET_BOUNDARIES 唯一决定，此标志仅为开关。
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    # use_time_features：从样本级 timestamp 派生 5 个请求时刻特征
    # (hour_sin, hour_cos, dow_sin, dow_cos, is_weekend)，追加到 user_dense_feats 末尾。
    # 开启后 user_dense_dim 自动扩大 5，模型 Dense FFN 输入维度随之调整，无需其他改动。
    parser.add_argument('--use_time_features', action='store_true', default=True,
                        help='Append 5 request-time features (hour_sin/cos, '
                             'dow_sin/cos, is_weekend) to user_dense_feats (default on)')
    parser.add_argument('--no_time_features', dest='use_time_features', action='store_false',
                        help='Disable request-time features')
    # use_time_ns：在保留连续时间特征的基础上，额外添加离散时间 NS token
    # （hour id + day-of-week id -> Embedding -> 1 个独立 time NS token）。
    parser.add_argument('--use_time_ns', action='store_true', default=True,
                        help='Enable one extra discrete time NS token (default on)')
    parser.add_argument('--no_time_ns', dest='use_time_ns', action='store_false',
                        help='Disable discrete time NS token')
    # rank_mixer_mode 控制 RankMixerBlock 的工作模式：
    #   full     — token mixing（跨 token 信息交换）+ per-token FFN，需满足 d_model % T == 0；
    #   ffn_only — 仅 per-token FFN，无 token mixing 约束；
    #   none     — 恒等直通，用于消融实验。
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # ── 5. 损失函数 ───────────────────────────────────────────────────────────
    # PCVR 正负样本极度不均衡时可切换为 Focal Loss，通过 alpha/gamma 压制易分负样本梯度。
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # ── 6. 稀疏优化器 & Embedding 构造控制 ───────────────────────────────────
    # Embedding 表使用 Adagrad 而非 AdamW：Adagrad 对稀疏梯度自适应缩放，
    # 适合工业推荐场景中每 step 只有少量 id 被激活的情况。
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    # reinit_sparse_after_epoch：从第 N 个 epoch 起，每轮结束后对高基数 Embedding 做冷重启，
    # 重置参数并重建 Adagrad 状态，防止高基数特征因记忆训练集而过拟合。
    # 参考：KuaiShou MultiEpoch（arxiv 2305.19531）。
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    # reinit_cardinality_threshold=0 表示禁用重初始化（不重置任何 Embedding）。
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # emb_skip_threshold：模型构造阶段跳过超高基数特征的 Embedding 分配，
    # 前向时以零向量替代，在不影响低基数特征精度的前提下节省 GPU 显存。
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    # seq_id_threshold：序列 tokenizer 内部区分"id 类特征"与"属性类特征"的阈值。
    # id 类特征（vocab_size > threshold）额外施加 2× dropout，降低对稀有 id 的记忆。
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    # ns_groups_json 默认指向项目根目录下的 ns_groups.json；
    # 若文件不存在，后续逻辑会退化为每个特征独立成组（singleton group）。
    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # ── NS Tokenizer 变体 ─────────────────────────────────────────────────────
    # group     — 每组特征经 FFN 投影为 1 个 NS Token，token 数 = 分组数；
    # rankmixer — 将所有 Embedding 拼接后均匀切分为固定数量的 token，
    #             token 数可调，无需 ns_groups.json，适合快速实验。
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    # user_ns_tokens / item_ns_tokens 仅在 rankmixer 模式下有效；
    # 设为 0 时自动使用分组数作为 token 数。
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()

    # 环境变量优先级高于 CLI，方便在训练平台/容器中统一注入路径而无需修改启动命令。
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # 确保输出目录存在，parents=True 支持多级目录一次创建。
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # 固定所有随机源（Python / NumPy / PyTorch CPU & CUDA），保证实验可复现。
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    # TensorBoard writer：训练 loss 和验证 AUC/LogLoss 均写入同一目录，
    # 方便用 tensorboard --logdir 实时监控训练曲线。
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    # schema.json 记录每个特征的 fid、offset、length 及词表大小，
    # 是构造 Embedding 表和切片 flat 输入向量的唯一依据。
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # 将 "seq_a:256,seq_b:256,seq_c:512,seq_d:512" 解析为 dict，
    # 传入 dataset 以对各行为域序列独立截断，减少长尾序列的 padding 浪费。
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    # get_pcvr_data 返回两个 DataLoader（train/valid）和一个 PCVRDataset 元信息对象。
    # valid_ratio 从尾部 Row Group 切分，与训练集严格不重叠。
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        use_time_features=args.use_time_features,
        use_time_ns=args.use_time_ns,
    )

    # ── NS 分组解析 ───────────────────────────────────────────────────────────
    #
    # 背景：HyFormer 的 Query Boosting 模块需要将所有输入特征表示成 T 个等长 token，
    # 才能执行 MLP-Mixer 风格的跨 token 混合（token mixing）。
    # 然而原始 NS（Non-Sequential）特征有几十个，不能直接一对一变成 token；
    # 需要先"分组压缩"，将语义相近的特征合并为同一个 NS token。
    #
    # ── 数据视角（从 Parquet 列到 flat tensor）─────────────────────────────
    #
    # Parquet 中每个 NS 整型特征以独立列存储，列名格式为 user_int_feats_{fid}。
    # dataset.py 的 _load_schema() 读取 schema.json（格式：[[fid, vocab_size, dim], ...]），
    # 按顺序将各特征拼接为一条 flat 向量，并在 FeatureSchema 中记录每个特征的位置：
    #
    #   user_int_feats  [B, total_dim]
    #   ├── fid=1   offset=0,  length=1   ← 单值整型特征（如用户性别）
    #   ├── fid=15  offset=1,  length=1
    #   ├── fid=48  offset=2,  length=4   ← 多值整型特征（如多热兴趣标签，长度=4）
    #   └── ...（共 N 个特征，总长 total_dim）
    #
    # ── 分组视角（ns_groups.json → NS token）────────────────────────────────
    #
    # ns_groups.json 将 fid 按语义聚合为若干"组"：
    #   "U1": [1, 15]        → 用户基础属性（如性别、年龄段）
    #   "U2": [48, 49, ...]  → 用户兴趣标签
    #   ...（共 7 个 user_int 组）
    #
    # GroupNSTokenizer（group 模式）对每组的处理流程：
    #   1. 按 fid 找到各特征在 user_int_feats 中的 offset，分别查对应 Embedding 表
    #   2. 将组内所有特征的 Embedding 向量拼接（concat）
    #   3. 经一个独立 FFN 投影为 d_model 维向量，即 1 个 NS token
    #
    # 因此 7 个 user_int 组 → 7 个 user_int NS token
    # + 1 个 user_dense token（所有 dense 特征拼接后统一投影）
    # + 4 个 item_int 组 → 4 个 item_int NS token
    # = 12 个 NS token（num_ns=12）
    #
    # 最终 T = num_queries × num_seq_domains + num_ns = 1×4 + 12 = 16
    # d_model=64 满足 64 % 16 == 0，符合 RankMixerBlock token mixing 的整除约束。
    #
    # ── fid → 本地下标的必要性 ──────────────────────────────────────────────
    #
    # ns_groups.json 存储的是业务侧 fid（如 1, 15, 48…），
    # 但模型内部通过 feature_specs 列表的下标（entry index，0-based）来定位
    # 每个特征的 Embedding 表和 flat tensor 切片位置。
    # 因此必须先建立 fid→entry_index 的映射，再把 JSON 中的 fid 替换为 entry index，
    # 才能传给模型构造器。
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        # fid → schema.entries 中的位置下标（entry index）
        # enumerate(schema.entries) 产生 (i, (fid, offset, length))，
        # 这里只取 fid 和 i，offset/length 通过 feature_specs 在模型内部使用。
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        # 将每组 fid 列表转换为对应的 entry index 列表；
        # 外层列表的长度 = 组数 = 最终产生的 NS token 数。
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        # 无分组配置时退化为 singleton group：每个特征独立成一组，产生一个 NS token。
        # token 数 = 特征数，通常远大于 group 模式，可能导致 T 较大进而违反 d_model % T == 0。
        # 适用于快速实验或无业务先验的场景，正式训练建议提供 ns_groups.json。
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ── 模型构建 ──────────────────────────────────────────────────────────────
    # build_feature_specs 将 schema 转换为 (vocab_size, offset, length) 三元组列表，
    # 供模型内部为每个特征分配独立的 Embedding 表并在 forward 时定位切片。
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        # 特征规格：整型 Embedding 表描述符 + Dense 维度
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        # 序列域：{domain_name: [vocab_size_per_feature_slot]}
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        # NS 分组：每组是一个特征本地下标列表
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        # Transformer 主干维度
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        # 序列编码器变体及相关参数
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        # 分类头
        "action_num": args.action_num,
        # num_time_buckets=0 时模型不分配时间桶 Embedding
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        # RankMixerBlock 工作模式
        "rank_mixer_mode": args.rank_mixer_mode,
        # RoPE 位置编码
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        # 超高基数特征跳过 Embedding（节省显存）
        "emb_skip_threshold": args.emb_skip_threshold,
        # 序列 id 特征额外 dropout 阈值
        "seq_id_threshold": args.seq_id_threshold,
        # NS Tokenizer 变体及 rankmixer 模式的 token 数
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        # 离散时间 NS token 开关（叠加于连续时间特征之上）
        "use_time_ns": args.use_time_ns,
        "query_pooling": args.query_pooling,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # 打印 token 数 T 的实际值，方便核验 RankMixerBlock 的 d_model % T == 0 约束。
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, "
                 f"rank_mixer_mode={args.rank_mixer_mode}, query_pooling={args.query_pooling}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ── 训练配置 ──────────────────────────────────────────────────────────────
    # EarlyStopping 监控验证集 AUC，连续 patience 次无改善则置位 early_stop 标志。
    # checkpoint_path 初始设为占位路径，trainer 在每次新 best 时动态更新为
    # global_stepN.best_model/model.pt 的实际路径。
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    # ckpt_params 会被编码进 checkpoint 目录名，
    # 格式：global_step{N}.layer={L}.head={H}.hidden={D}[.best_model]
    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        # ns_groups_path 只在文件实际存在时传入，使 checkpoint 目录自包含：
        # trainer 会将 ns_groups.json 复制进每个 best_model 目录，
        # 推理时无需依赖训练机器上的原始路径。
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        # 完整超参快照写入 checkpoint 的 train_config.json，便于复现和审计
        train_config=vars(args),
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
