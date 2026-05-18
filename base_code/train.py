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
from typing import Dict, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    NUM_TIME_BUCKETS,
    SAMPLE_TIME_DIM,
    SAMPLE_TIME_BUCKET_SIZES,
    SEQ_TIME_DENSE_DIM,
    SEQ_TIME_BUCKET_SIZES,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "default_run"


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_int_list(value: str) -> List[int]:
    """Parse comma-separated integers, returning an empty list for blank input."""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def build_user_pair_specs(
    schema: FeatureSchema,
    dense_schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
    fids: List[int],
) -> List[Dict[str, int]]:
    """Build specs for aligned ``(user_int_id, user_dense_weight)`` features."""
    specs: List[Dict[str, int]] = []
    for fid in fids:
        int_offset, int_len = schema.get_offset_length(fid)
        dense_offset, dense_len = dense_schema.get_offset_length(fid)
        if int_len != dense_len:
            raise ValueError(
                f"user pair fid={fid} has mismatched int_len={int_len} "
                f"and dense_len={dense_len}")
        vocab_size = max(per_position_vocab_sizes[int_offset:int_offset + int_len])
        specs.append({
            "fid": fid,
            "vocab_size": int(vocab_size),
            "int_offset": int(int_offset),
            "dense_offset": int(dense_offset),
            "length": int(int_len),
        })
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str,
                        default=str(PROJECT_DIR / 'data' / 'demo_prepared'),
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str,
                        default=str(DEFAULT_OUTPUT_DIR / 'checkpoints'),
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str,
                        default=str(DEFAULT_OUTPUT_DIR / 'logs'),
                        help='Log directory (env: TRAIN_LOG_PATH)')
    parser.add_argument('--tf_events_dir', type=str, default=None,
                        help='TensorBoard event directory '
                             '(env: TRAIN_TF_EVENTS_PATH; defaults to <log_dir>/tensorboard)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
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

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
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
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
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

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # User pair features: aligned user_int/user_dense fids 62-66.
    parser.add_argument('--use_user_pair', action='store_true', default=True,
                        help='Enable aligned (user_int, user_dense) pair encoder '
                             'for user fids such as 62-66')
    parser.add_argument('--no_user_pair', dest='use_user_pair', action='store_false',
                        help='Disable user pair encoder')
    parser.add_argument('--user_pair_fids', type=str, default='62,63,64,65,66',
                        help='Comma-separated user fids whose int/dense arrays are aligned')
    parser.add_argument('--user_pair_emb_dim', type=int, default=32,
                        help='Embedding dimension used inside UserPairEncoder')
    parser.add_argument('--user_pair_use_rank', action='store_true', default=True,
                        help='Add per-slot rank embeddings inside UserPairEncoder')
    parser.add_argument('--no_user_pair_rank', dest='user_pair_use_rank', action='store_false',
                        help='Disable per-slot rank embeddings inside UserPairEncoder')
    parser.add_argument('--user_pair_item_aware', action='store_true', default=True,
                        help='Add an item-aware user pair token')
    parser.add_argument('--no_user_pair_item_aware', dest='user_pair_item_aware',
                        action='store_false', help='Disable item-aware user pair token')
    parser.add_argument('--user_pair_time_aware', action='store_true', default=True,
                        help='Condition the item-aware user pair token on sample time')
    parser.add_argument('--no_user_pair_time_aware', dest='user_pair_time_aware',
                        action='store_false', help='Disable sample-time conditioning for user pair')
    parser.add_argument('--user_pair_use_hash_cross', action='store_true', default=False,
                        help='Add top-k hash-cross user pair token')
    parser.add_argument('--user_pair_hash_bucket_size', type=int, default=200000,
                        help='Hash bucket size for optional user pair hash cross')
    parser.add_argument('--user_pair_hash_top_k', type=int, default=3,
                        help='Top-k slots per fid used by optional user pair hash cross')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH', args.tf_events_dir)
    if args.tf_events_dir is None:
        args.tf_events_dir = os.path.join(args.log_dir, 'tensorboard')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
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
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)
    user_pair_fids = _parse_int_list(args.user_pair_fids)
    user_pair_specs = (
        build_user_pair_specs(
            pcvr_dataset.user_int_schema,
            pcvr_dataset.user_dense_schema,
            pcvr_dataset.user_int_vocab_sizes,
            user_pair_fids,
        )
        if args.use_user_pair and user_pair_fids
        else []
    )
    if user_pair_specs:
        logging.info(f"User pair specs: {user_pair_specs}")

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "sample_time_dim": SAMPLE_TIME_DIM,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "sample_time_bucket_sizes": SAMPLE_TIME_BUCKET_SIZES,
        "seq_time_dense_dim": SEQ_TIME_DENSE_DIM,
        "seq_time_bucket_sizes": SEQ_TIME_BUCKET_SIZES,
        "user_pair_specs": user_pair_specs,
        "user_pair_emb_dim": args.user_pair_emb_dim,
        "user_pair_use_rank": args.user_pair_use_rank,
        "user_pair_item_aware": args.user_pair_item_aware,
        "user_pair_time_aware": args.user_pair_time_aware,
        "user_pair_use_hash_cross": args.user_pair_use_hash_cross,
        "user_pair_hash_bucket_size": args.user_pair_hash_bucket_size,
        "user_pair_hash_top_k": args.user_pair_hash_top_k,
    }

    if args.ns_tokenizer_type == 'rankmixer':
        num_user_ns = args.user_ns_tokens if args.user_ns_tokens > 0 else len(user_ns_groups)
        num_item_ns = args.item_ns_tokens if args.item_ns_tokens > 0 else len(item_ns_groups)
    else:
        num_user_ns = len(user_ns_groups)
        num_item_ns = len(item_ns_groups)
    estimated_num_ns = (
        num_user_ns
        + (1 if pcvr_dataset.user_dense_schema.total_dim > 0 else 0)
        + num_item_ns
        + (1 if pcvr_dataset.item_dense_schema.total_dim > 0 else 0)
        + (1 if SAMPLE_TIME_DIM > 0 else 0)
    )
    if user_pair_specs:
        estimated_num_ns += 1
        if args.user_pair_item_aware or args.user_pair_time_aware:
            estimated_num_ns += 1
        if args.user_pair_use_hash_cross:
            estimated_num_ns += 1
    estimated_T = args.num_queries * len(pcvr_dataset.seq_domains) + estimated_num_ns
    if model_args["rank_mixer_mode"] == 'full' and args.d_model % estimated_T != 0:
        logging.warning(
            "rank_mixer_mode=full requires d_model %% T == 0, but "
            f"d_model={args.d_model}, T={estimated_T} after adding the "
            "sample-time token. Falling back to rank_mixer_mode=ffn_only. "
            "To keep full RankMixer, choose a compatible --d_model or token count."
        )
        model_args["rank_mixer_mode"] = 'ffn_only'
        args.rank_mixer_mode = 'ffn_only'

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

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
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
