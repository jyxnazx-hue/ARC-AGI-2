"""
training loop for TRM on ARC-AGI-2.

Run modes:
    python train_trm.py --config smoke    # 3 tasks, 3 steps, tiny model, ~20s CPU
    python train_trm.py --config real     # 800 tasks, 3000 steps, full model, L4 GPU

Features:
    * AdamW with two param groups: model at lr, puzzle embeddings at 10x lr
    * Linear warmup + cosine decay
    * Gradient accumulation
    * EMA of weights
    * Gradient clipping
    * Holdout pass@1 eval every N steps (single forward, argmax over color tokens)
    * Checkpoint save/load (model + optimizer + scheduler + EMA + step)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# pyrefly: ignore[missing-import]
import numpy as np

# pyrefly: ignore[missing-import]
import torch

# pyrefly: ignore[missing-import]
import torch.nn as nn

from src.core import dataset as ds
from src.core import tokenizer as tk
from src.models.trm_core import TRMModel

# Configs

@dataclass
class Config:
    name: str = "smoke"

    # Paths
    data_root: Path = Path("data")
    train_split: str = "build"
    holdout_split: str = "holdout"
    ckpt_dir: Path = Path("checkpoints")

    # Data
    n_train_pairs: int = 4          # 4 train + 1 test = 5 canvases per side
    n_color_perms: int = 2          # 24 variants per task
    max_train_tasks: Optional[int] = None
    max_holdout_tasks: Optional[int] = 50

    # Model
    vocab_size: int = 12
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    ffn_dim: int = 1024
    inner_steps_k: int = 4
    outer_steps_t: int = 3
    n_puzzles: int = 800
    puzzle_len: int = 8

    # Optim
    lr_model: float = 1e-4
    lr_puzzle: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Schedule
    warmup_steps: int = 100
    total_steps: int = 3000
    min_lr_ratio: float = 0.1

    # EMA
    ema_decay: float = 0.9999

    # Training
    batch_size: int = 8
    tbptt_window: int = 4
    dtype: str = "fp32"             # "fp32" or "bf16"
    seed: int = 0

    # Logging / checkpointing
    log_every: int = 10
    eval_every: int = 500
    ckpt_every: int = 250          
    milestone_every: int = 500     
    n_eval_tasks: int = 50

    time_budget_min: float = 240.0 
    time_buffer_min: float = 5.0    

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


SMOKE = Config(
    name="smoke",
    max_train_tasks=3,
    max_holdout_tasks=3,
    n_color_perms=0,
    d_model=32, n_heads=2, n_layers=1, ffn_dim=64,
    inner_steps_k=2, outer_steps_t=2,
    n_puzzles=16, puzzle_len=4,
    total_steps=3, warmup_steps=1, log_every=1, eval_every=2,
    ckpt_every=3, milestone_every=3,
    batch_size=2, n_eval_tasks=3,
    time_budget_min=5.0, time_buffer_min=0.5,
    ckpt_dir=Path("checkpoints_smoke"),
)

REAL = Config(
    name="real",
    batch_size=2,               
    grad_accum_steps=8,
    total_steps=1500,         
    ckpt_every=100,             
    milestone_every=250,
    dtype="fp32",               
    time_budget_min=480.0,     
    n_eval_tasks=50,
)


# Utilities

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cos


class EMA:
    """Exponential moving average of model parameters (in-place update)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        for n in self.shadow:
            self.shadow[n].copy_(sd[n])

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.copy_(self.shadow[n])


# Model / data factory

def build_model(cfg: Config) -> TRMModel:
    return TRMModel(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ffn_dim=cfg.ffn_dim,
        inner_steps_k=cfg.inner_steps_k,
        outer_steps_t=cfg.outer_steps_t,
        n_puzzles=cfg.n_puzzles,
        puzzle_len=cfg.puzzle_len,
        n_pairs=cfg.n_train_pairs + 1,
    )


def build_datasets(cfg: Config):
    data_root = Path(cfg.data_root)
    train_dir = data_root / cfg.train_split
    holdout_dir = data_root / cfg.holdout_split

    train_paths = sorted(train_dir.glob("*.json"))
    if cfg.max_train_tasks is not None:
        train_paths = train_paths[: cfg.max_train_tasks]
    if not train_paths:
        raise FileNotFoundError(f"no tasks in {train_dir}")

    # puzzle_idx map is defined over the *training* set
    puzzle_idx_map = ds.build_puzzle_idx_map(train_paths)

    train_ds = ds.ARCTrainDataset(
        train_paths, puzzle_idx_map,
        n_train_pairs=cfg.n_train_pairs,
        n_color_perms=cfg.n_color_perms,
        augment=True, seed=cfg.seed,
    )

    holdout_ds = None
    if holdout_dir.exists():
        holdout_paths = sorted(holdout_dir.glob("*.json"))
        if cfg.max_holdout_tasks is not None:
            holdout_paths = holdout_paths[: cfg.max_holdout_tasks]
        if holdout_paths:
            all_paths = train_paths + holdout_paths
            full_map = ds.build_puzzle_idx_map(all_paths)
            holdout_ds = ds.ARCTrainDataset(
                holdout_paths, full_map,
                n_train_pairs=cfg.n_train_pairs,
                n_color_perms=0, augment=False,
            )
    return train_ds, holdout_ds, puzzle_idx_map


def build_optimizer(model: TRMModel, cfg: Config):
    """Two param groups: everything except puzzle table at lr_model,
    puzzle table at lr_puzzle (typically 10x)."""
    puzzle_name = "puzzle_embeddings.weight"
    model_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n != puzzle_name]
    puzzle_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and n == puzzle_name]

    param_groups = [
        {"params": model_params, "lr": cfg.lr_model, "name": "model"},
        {"params": puzzle_params, "lr": cfg.lr_puzzle, "name": "puzzle"},
    ]
    return torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

# Evaluation — single forward pass, pass@1

def _tensor_batch_of_one(sample: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        elif isinstance(v, int):
            out[k] = torch.tensor([v], dtype=torch.long)
        else:
            out[k] = [v]
    return out


@torch.no_grad()
def evaluate_pass_at_1(
    model: TRMModel,
    eval_ds: ds.ARCTrainDataset,
    cfg: Config,
    n_tasks: int,
) -> float:
    model.eval()
    n_total = min(n_tasks, len(eval_ds))
    n_correct = 0

    for i in range(n_total):
        sample = eval_ds[i]

        test_cell_mask = sample["output_cell_mask"][-1].bool()  # [900]
        masked_test_out = torch.where(
            test_cell_mask,
            torch.full_like(sample["output_canvases"][-1], tk.MASK),
            torch.zeros_like(sample["output_canvases"][-1]),
        )
        sample["output_canvases"] = sample["output_canvases"].clone()
        sample["output_canvases"][-1] = masked_test_out

        batch = _tensor_batch_of_one(sample)
        token_ids, attention, _ = ds.build_sequence(batch, puzzle_len=cfg.puzzle_len)

        token_ids = token_ids.to(cfg.device)
        attention = attention.to(cfg.device)
        puzzle_idx = batch["puzzle_idx"].to(cfg.device)

        out = model(token_ids, puzzle_idx, attention_mask=attention, loss_mask=None)
        test_logits = out["final_logits"][0, -tk.CANVAS_SIZE:]  # [900, 12]
        pred = test_logits[..., 2:12].argmax(dim=-1) + 2        # [900]

        truth = sample["output_canvases"][-1]                   # [900]
        non_pad = truth != tk.PAD
        if non_pad.sum() == 0:
            n_correct += 1
            continue
        if torch.equal(pred[non_pad], truth[non_pad]):
            n_correct += 1

    model.train()
    return n_correct / max(1, n_total)


# Trainer

class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)

        self.train_ds, self.holdout_ds, self.puzzle_idx_map = build_datasets(cfg)
        self.model = build_model(cfg).to(cfg.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: lr_lambda(s, cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio),
        )
        self.ema = EMA(self.model, cfg.ema_decay)

        self.step = 0
        self.dataloader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=cfg.batch_size,
            shuffle=True, collate_fn=ds.collate_fn,
            num_workers=0, drop_last=False,
        )
        self._dl_iter = None

        self.start_time: Optional[float] = None
        self.step_at_start: int = 0
        self.best_holdout: float = -1.0

        self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _next_batch(self):
        if self._dl_iter is None:
            self._dl_iter = iter(self.dataloader)
        try:
            return next(self._dl_iter)
        except StopIteration:
            self._dl_iter = iter(self.dataloader)
            return next(self._dl_iter)

    def train_step(self) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.cfg.grad_accum_steps)
        total_loss = 0.0

        for _ in range(accum):
            batch = self._next_batch()
            token_ids, attention, loss_mask = ds.build_sequence(
                batch, puzzle_len=self.cfg.puzzle_len
            )
            token_ids = token_ids.to(self.cfg.device)
            attention = attention.to(self.cfg.device)
            loss_mask = loss_mask.to(self.cfg.device)
            puzzle_idx = batch["puzzle_idx"].to(self.cfg.device)

            if self.cfg.dtype == "bf16":
                with torch.autocast(device_type=self.cfg.device.split(":")[0],
                                    dtype=torch.bfloat16):
                    out = self.model(
                        token_ids, puzzle_idx,
                        attention_mask=attention, loss_mask=loss_mask,
                        tbptt_window=self.cfg.tbptt_window,
                    )
                    loss = out["total_loss"]
            else:
                out = self.model(
                    token_ids, puzzle_idx,
                    attention_mask=attention, loss_mask=loss_mask,
                    tbptt_window=self.cfg.tbptt_window,
                )
                loss = out["total_loss"]

            (loss / accum).backward()
            total_loss += loss.item() / accum

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update(self.model)
        self.step += 1

        return {
            "loss": total_loss,
            "lr_model": self.optimizer.param_groups[0]["lr"],
            "lr_puzzle": self.optimizer.param_groups[1]["lr"],
        }

    def save_checkpoint(self, tag: str) -> Path:
        path = self.cfg.ckpt_dir / f"trm_{tag}.pt"
        torch.save({
            "config_name": self.cfg.name,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "puzzle_idx_map": self.puzzle_idx_map,
        }, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.ema.load_state_dict(ckpt["ema"])
        self.step = ckpt["step"]

    def _estimate_eta_sec(self) -> Optional[float]:
        if self.start_time is None:
            return None
        steps_done = self.step - self.step_at_start
        if steps_done < 5:
            return None
        elapsed = time.time() - self.start_time
        sec_per_step = elapsed / steps_done
        remaining = max(0, self.cfg.total_steps - self.step)
        return remaining * sec_per_step

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = int(seconds)
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    def _budget_exceeded(self) -> bool:
        if self.start_time is None or self.cfg.time_budget_min <= 0:
            return False
        elapsed_min = (time.time() - self.start_time) / 60.0
        remaining_min = self.cfg.time_budget_min - elapsed_min
        if remaining_min < self.cfg.time_buffer_min:
            return True
        eta = self._estimate_eta_sec()
        if eta is not None and eta / 60.0 > (remaining_min - self.cfg.time_buffer_min):
            return True
        return False

    def _checkpoint(self, tag: str, log: bool = True) -> Path:
        p = self.save_checkpoint(tag)
        if log:
            print(f"[ckpt] saved {p}")
        return p

    def train(self) -> None:
        cfg = self.cfg
        self.start_time = time.time()
        self.step_at_start = self.step

        print(f"[train] config={cfg.name}  device={cfg.device}  "
              f"params={count_parameters(self.model):,}")
        print(f"[train] train rows={len(self.train_ds):,}  "
              f"holdout tasks={len(self.holdout_ds) if self.holdout_ds else 0}  "
              f"steps={cfg.total_steps}  budget={cfg.time_budget_min:.0f}min")

        recent_losses: List[float] = []
        stopped_reason = "completed"

        try:
            while self.step < cfg.total_steps:
                stats = self.train_step()
                recent_losses.append(stats["loss"])

                if self.step % cfg.log_every == 0:
                    avg = sum(recent_losses) / len(recent_losses)
                    recent_losses = []
                    elapsed = time.time() - self.start_time
                    eta = self._estimate_eta_sec()
                    eta_str = self._fmt_duration(eta) if eta else "?"
                    print(
                        f"[step {self.step:5d}/{cfg.total_steps}] "
                        f"loss={avg:.4f}  "
                        f"lr_model={stats['lr_model']:.2e}  "
                        f"lr_puzzle={stats['lr_puzzle']:.2e}  "
                        f"elapsed={self._fmt_duration(elapsed)}  "
                        f"eta={eta_str}"
                    )

                if self.step % cfg.ckpt_every == 0:
                    self._checkpoint("latest")
                if self.step % cfg.milestone_every == 0:
                    self._checkpoint(f"step{self.step}")

                if self.holdout_ds and self.step % cfg.eval_every == 0:
                    acc = evaluate_pass_at_1(
                        self.model, self.holdout_ds, cfg, cfg.n_eval_tasks
                    )
                    marker = ""
                    if acc > self.best_holdout:
                        self.best_holdout = acc
                        self._checkpoint("best", log=False)
                        marker = "  (saved best.pt)"
                    print(f"[eval  step {self.step}] holdout pass@1 = {acc:.3f}{marker}")

                if self._budget_exceeded():
                    stopped_reason = "time_budget"
                    print(f"[stop] time budget reached at step {self.step} "
                          f"({self._fmt_duration(time.time() - self.start_time)})")
                    break

        except KeyboardInterrupt:
            stopped_reason = "interrupted"
            print(f"\n[stop] KeyboardInterrupt at step {self.step}")
            self._checkpoint("interrupt")
        except Exception as e:
            stopped_reason = f"error: {type(e).__name__}"
            print(f"\n[stop] exception at step {self.step}: {e}")
            self._checkpoint("interrupt")
            raise

        self._checkpoint("final")
        total = time.time() - self.start_time
        print(f"[done] reason={stopped_reason}  steps={self.step}  "
              f"total_time={self._fmt_duration(total)}  "
              f"best_holdout={self.best_holdout:.3f}")

# Entry point

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["smoke", "real"], default="smoke")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = SMOKE if args.config == "smoke" else REAL
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    trainer.train()


if __name__ == "__main__":
    main()
