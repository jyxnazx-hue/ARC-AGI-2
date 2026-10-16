from __future__ import annotations

from pathlib import Path

# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import pytest

from train_trm import (
    SMOKE, Config, Trainer, EMA, lr_lambda, build_model, build_datasets,
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = Config(**{**SMOKE.__dict__})
    c.ckpt_dir = tmp_path / "ckpt"
    return c


# Smoke

def test_smoke_train_runs(cfg):
    trainer = Trainer(cfg)
    trainer.train()
    assert trainer.step == cfg.total_steps
    assert (cfg.ckpt_dir / "trm_final.pt").exists()


def test_smoke_step_returns_loss(cfg):
    trainer = Trainer(cfg)
    stats = trainer.train_step()
    assert "loss" in stats
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_smoke_loss_is_finite_over_three_steps(cfg):
    trainer = Trainer(cfg)
    losses = [trainer.train_step()["loss"] for _ in range(3)]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)

# EMA

def test_ema_shadow_is_copy_initial(cfg):
    m = build_model(cfg)
    ema = EMA(m, decay=0.9)
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert torch.allclose(ema.shadow[n], p)


def test_ema_updates_toward_new_params(cfg):
    m = build_model(cfg)
    ema = EMA(m, decay=0.9)
    before = ema.shadow["token_embeddings.weight"].clone()

    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)

    after = ema.shadow["token_embeddings.weight"]
    delta = after - before
    assert (delta.abs() > 1e-6).any()


# LR schedule

def test_lr_warmup_starts_small():
    assert lr_lambda(0, warmup=10, total=100, min_ratio=0.1) == 0.0
    assert lr_lambda(5, warmup=10, total=100, min_ratio=0.1) == 0.5


def test_lr_decays_after_peak():
    peak = lr_lambda(10, warmup=10, total=100, min_ratio=0.1)
    late = lr_lambda(90, warmup=10, total=100, min_ratio=0.1)
    assert peak > late
    assert late >= 0.1

def test_checkpoint_roundtrip(cfg):
    trainer = Trainer(cfg)
    trainer.train_step()
    path = trainer.save_checkpoint("test")
    assert path.exists()

    trainer2 = Trainer(cfg)
    trainer2.load_checkpoint(path)
    assert trainer2.step == trainer.step

    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(),
        trainer2.model.named_parameters(),
    ):
        assert torch.allclose(p1, p2), f"mismatch on {n1}"