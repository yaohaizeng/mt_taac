import json
import os
import random
import copy
import logging
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter:
    """Custom ``logging.Formatter`` that prefixes every record with the
    wall-clock timestamp and the elapsed wall-clock time since this
    formatter instance was constructed.

    The prefix format is ``"<locale-date> <locale-time> - H:MM:SS"``, which
    is convenient for tracking long-running training runs where both the
    absolute time and the time-since-start are useful.

    Multi-line messages are re-indented so that continuation lines align
    with the beginning of the message (not the prefix).
    """

    def __init__(self) -> None:
        # Anchor used to compute the elapsed-time part of the log prefix.
        # Can be reset at runtime via ``create_logger(...).reset_time()``.
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # Indent continuation lines so they line up with the message body,
        # not with the timestamp prefix.
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """Create and configure the root logger for a training/inference run.

    The returned logger has two handlers attached:

    * A ``FileHandler`` bound to ``filepath`` (opened in write mode,
      truncating any previous content) that records ``DEBUG``-level and
      above messages for post-mortem inspection.
    * A ``StreamHandler`` to stderr that only echoes ``INFO``-level and
      above messages, keeping the console output concise.

    Both handlers share a ``LogFormatter`` so the console and the log file
    stay in sync. Any pre-existing handlers on the root logger are removed
    to avoid duplicate lines when this function is called multiple times.

    Args:
        filepath: Destination path of the log file. Opened in ``"w"`` mode,
            so previous contents are overwritten.

    Returns:
        The root ``logging.Logger`` instance. The returned object is
        augmented with a ``reset_time()`` attribute that resets the
        elapsed-time clock used by the log prefix. This is useful when the
        "interesting" phase of a run starts well after process launch
        (e.g. after schema building and data loading).
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Allow callers to reset the elapsed-time clock shown in the log prefix.
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """Early-stop training when the validation metric plateaus.

    The tracker assumes a *higher-is-better* metric (typical for AUC or
    accuracy). A candidate ``score`` is considered an improvement iff
    ``score > best_score + delta``; otherwise the internal ``counter`` is
    incremented and training is requested to stop once
    ``counter >= patience``.

    On every improvement the current ``model.state_dict()`` is both
    deep-copied in memory (``self.best_model``) and persisted to disk at
    ``checkpoint_path``. The most recent *improving* score is cached in
    ``self.best_saved_score`` so callers can skip redundant IO.

    Attributes:
        checkpoint_path: Destination path for the best ``state_dict``.
        patience: Number of non-improving calls tolerated before
            ``early_stop`` is flipped to ``True``.
        verbose: If ``True``, emit an ``INFO`` line whenever a checkpoint
            is written.
        counter: Number of consecutive non-improving calls seen so far.
        best_score: Best score observed; ``None`` until the first call.
        early_stop: Set to ``True`` once ``counter >= patience``.
        delta: Minimum absolute improvement required to reset ``counter``.
        best_model: In-memory deep copy of the best ``state_dict``.
        best_saved_score: Score associated with the last checkpoint
            actually written to disk.
        best_extra_metrics: Optional auxiliary metrics captured at the
            best-score step (e.g. logloss, other AUCs).
        label: Short prefix (e.g. ``"val"``) prepended to log lines to
            disambiguate multiple trackers running in parallel.
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """Return ``True`` iff ``score`` fails to beat ``best_score + delta``.

        Used as the gating condition for incrementing the patience counter.
        ``best_score`` must have been seeded by a prior ``__call__``.
        """
        assert self.best_score is not None, "call __call__ first to seed best_score"
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed a new validation score into the tracker.

        Three branches, in order:

        1. First call (``best_score is None``): seed the tracker, persist a
           checkpoint, and cache the model weights.
        2. Not improved: increment ``counter`` and log the progress; flip
           ``early_stop`` once ``counter >= patience``.
        3. Improved: reset ``counter`` to ``0``, update ``best_score`` and
           ``best_extra_metrics``, refresh the in-memory ``best_model``,
           and write a new checkpoint to disk.

        Args:
            score: Scalar validation metric (higher is better, e.g. AUC).
            model: Model whose ``state_dict`` is snapshotted on
                improvement. Only the parameters are saved, not the
                optimizer state.
            extra_metrics: Optional dict of auxiliary metrics recorded at
                the same step, e.g.
                ``{"best_val_AUC": ..., "best_val_logloss": ...}``. Stored
                verbatim as ``self.best_extra_metrics``; not interpreted
                by ``EarlyStopping`` itself.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """Persist ``model.state_dict()`` to ``self.checkpoint_path``.

        Creates any missing parent directories, writes atomically via
        ``torch.save``, and records ``score`` as ``self.best_saved_score``
        so subsequent callers can detect "no new improvement since last
        save" without re-reading the checkpoint file.

        Args:
            score: Validation score associated with the weights being
                saved. Exposed to callers via ``best_saved_score`` after
                the write completes.
            model: Model whose parameters are being snapshotted. Only
                ``state_dict()`` is written; optimizer and scheduler state
                are explicitly *not* included.
        """
        if self.verbose:
            logging.info('Validation score increased. Saving model ...')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_saved_score = score


_DATA_TAG = '[DATA]'


def log_schema_for_eda(
    schema_path: str,
    log_dir: Optional[str] = None,
    log_full_json: bool = False,
) -> None:
    """Log schema summary (and optionally full JSON) for platform-side EDA.

    Always writes ``schema_dump.json`` under ``log_dir`` when provided.
    """
    with open(schema_path, 'r', encoding='utf-8') as f:
        raw: Dict[str, Any] = json.load(f)

    lines: List[str] = [
        f'{_DATA_TAG} === schema summary ===',
        f'{_DATA_TAG} schema_path: {schema_path}',
    ]
    for key in ('user_int', 'item_int'):
        cols = raw.get(key, [])
        if not cols:
            continue
        fids = [c[0] for c in cols]
        max_vocab = max(c[1] for c in cols)
        total_dim = sum(c[2] for c in cols)
        lines.append(
            f'{_DATA_TAG}   {key}: n_features={len(cols)}, flat_dim={total_dim}, '
            f'max_vocab={max_vocab}, fids={fids}')
    user_dense = raw.get('user_dense', [])
    if user_dense:
        lines.append(
            f'{_DATA_TAG}   user_dense: n_features={len(user_dense)}, '
            f'fids={[c[0] for c in user_dense]}')
    seq_cfg = raw.get('seq', {})
    for domain in sorted(seq_cfg.keys()):
        cfg = seq_cfg[domain]
        feats = cfg.get('features', [])
        lines.append(
            f'{_DATA_TAG}   seq.{domain}: prefix={cfg.get("prefix")}, '
            f'ts_fid={cfg.get("ts_fid")}, n_side_feats={len(feats)}, '
            f'fids_vocab={[(f, v) for f, v in feats]}')
    logging.info('\n'.join(lines))

    payload = json.dumps(raw, ensure_ascii=False, indent=2)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        dump_path = os.path.join(log_dir, 'schema_dump.json')
        with open(dump_path, 'w', encoding='utf-8') as f:
            f.write(payload)
        logging.info(f'{_DATA_TAG} Full schema copied to {dump_path}')

    if log_full_json:
        logging.info(f'{_DATA_TAG} schema.json begin >>>')
        for line in payload.splitlines():
            logging.info(f'{_DATA_TAG} {line}')
        logging.info(f'{_DATA_TAG} schema.json end <<<')


def log_batch_data_stats(batch: Dict[str, Any], tag: str = 'train') -> None:
    """Log one-batch data statistics (label rate, seq lengths, padding, time)."""
    lines: List[str] = [f'{_DATA_TAG} === batch stats ({tag}) ===']

    label = batch.get('label')
    if isinstance(label, torch.Tensor):
        y = label.detach().float().cpu()
        lines.append(
            f'{_DATA_TAG}   label: pos_rate={y.mean().item():.6f}, '
            f'n_pos={int(y.sum().item())}, batch_size={y.numel()}')

    ts = batch.get('timestamp')
    if isinstance(ts, torch.Tensor):
        ts_np = ts.detach().cpu().numpy()
        lines.append(
            f'{_DATA_TAG}   timestamp: min={int(ts_np.min())}, max={int(ts_np.max())}')

    for key in ('user_int_feats', 'item_int_feats'):
        t = batch.get(key)
        if isinstance(t, torch.Tensor):
            x = t.detach().cpu()
            zero_frac = (x == 0).float().mean().item()
            lines.append(
                f'{_DATA_TAG}   {key}: shape={list(x.shape)}, '
                f'zero_frac={zero_frac:.4f}, max_id={int(x.max().item())}')

    ud = batch.get('user_dense_feats')
    if isinstance(ud, torch.Tensor) and ud.numel() > 0:
        u = ud.detach().float().cpu()
        lines.append(
            f'{_DATA_TAG}   user_dense_feats: shape={list(u.shape)}, '
            f'mean={u.mean().item():.4f}, std={u.std().item():.4f}')

    seq_domains = batch.get('_seq_domains', [])
    for domain in seq_domains:
        lens_key = f'{domain}_len'
        lens = batch.get(lens_key)
        if isinstance(lens, torch.Tensor):
            l = lens.detach().cpu().float()
            lines.append(
                f'{_DATA_TAG}   {domain}_len: mean={l.mean().item():.2f}, '
                f'max={int(l.max().item())}, min={int(l.min().item())}')
        max_len = None
        seq_t = batch.get(domain)
        if isinstance(seq_t, torch.Tensor):
            max_len = seq_t.shape[2]
        if max_len is not None and isinstance(lens, torch.Tensor):
            truncated = (lens > max_len).sum().item()
            if truncated:
                lines.append(
                    f'{_DATA_TAG}   {domain}: truncated_rows={truncated} '
                    f'(len > max_len={max_len})')

    logging.info('\n'.join(lines))


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training reproducibility.

    Seeds ``random``, the ``PYTHONHASHSEED`` env var, NumPy, the CPU
    PyTorch generator and all CUDA generators, then forces cuDNN into
    deterministic mode.

    Note that full bitwise determinism on GPU also requires disabling
    cuDNN auto-tuning (``torch.backends.cudnn.benchmark = False``) and may
    come with a non-trivial throughput cost; this helper intentionally
    only toggles ``deterministic`` to preserve speed for common use cases.

    Args:
        seed: Non-negative integer seed shared by all RNGs listed above.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N,) raw logits (before sigmoid).
        targets: (N,) binary labels {0, 1}.
        alpha: positive-class weight in (0, 1). When positives dominate,
            use alpha < 0.5 to downweight the positive class.
        gamma: focusing parameter. gamma=0 degenerates to standard BCE;
            gamma=2 is the standard value.
        reduction: 'mean' | 'sum' | 'none'.
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
