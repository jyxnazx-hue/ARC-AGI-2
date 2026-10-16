from __future__ import annotations

# pyrefly: ignore[missing-import]
import torch 
# pyrefly: ignore[missing-import]
import pytest

from src.models.trm_core import TRMModel

TINY = dict(
    vocab_size=12, d_model=32, n_heads=2, n_layers=1, ffn_dim=64,
    inner_steps_k=2, outer_steps_t=1,
    n_puzzles=4, puzzle_len=2, n_pairs=2,
    canvas_h=4, canvas_w=4,
)
TINY_T = TINY["puzzle_len"] + TINY["n_pairs"] * 2 * (TINY["canvas_h"] * TINY["canvas_w"])
B = 2


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TRMModel(**TINY)


def _batch(puzzle_idx=None):
    torch.manual_seed(0)
    token_ids = torch.randint(0, 12, (B, TINY_T))
    if puzzle_idx is None:
        puzzle_idx = torch.zeros(B, dtype=torch.long)
    attention_mask = torch.ones(B, TINY_T, dtype=torch.bool)
    loss_mask = torch.zeros(B, TINY_T, dtype=torch.bool)
    loss_mask[:, -16:] = True   # last canvas = test output
    return token_ids, puzzle_idx, attention_mask, loss_mask


# Manual-computation check

def test_loss_matches_manual_cross_entropy(model):
    """With outer_steps_t=1, total_loss == CE(final_logits, labels)."""
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    out = model(token_ids, puzzle_idx,
                attention_mask=attention_mask, loss_mask=loss_mask)

    labels = torch.where(loss_mask, token_ids, -100)
    manual = torch.nn.functional.cross_entropy(
        out["final_logits"].reshape(-1, 12),
        labels.reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out["total_loss"], manual, atol=1e-6)


# Masking behaviour

def test_loss_ignores_pad_positions(model):
    """Changing tokens at pad/ignore positions should not change loss."""
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    torch.manual_seed(7)
    base = model(token_ids, puzzle_idx,
                 attention_mask=attention_mask, loss_mask=loss_mask)

    ids = token_ids.clone()
    ids[:, 0] = (ids[:, 0] + 1) % 12   # first puzzle token, not in loss
    torch.manual_seed(7)
    out = model(ids, puzzle_idx,
                attention_mask=attention_mask, loss_mask=loss_mask)

    assert torch.allclose(base["total_loss"], out["total_loss"], atol=1e-6)


def test_loss_changes_when_target_changes(model):
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    torch.manual_seed(7)
    base = model(token_ids, puzzle_idx,
                 attention_mask=attention_mask, loss_mask=loss_mask)

    ids = token_ids.clone()
    ids[:, -1] = (ids[:, -1] + 1) % 12   # last token, in loss mask
    torch.manual_seed(7)
    out = model(ids, puzzle_idx,
                attention_mask=attention_mask, loss_mask=loss_mask)

    assert not torch.allclose(base["total_loss"], out["total_loss"])


def test_loss_is_zero_when_mask_empty(model):
    """All-ignore labels must give loss 0 (not NaN)."""
    token_ids, puzzle_idx, attention_mask, _ = _batch()
    empty_mask = torch.zeros(B, TINY_T, dtype=torch.bool)

    out = model(token_ids, puzzle_idx,
                attention_mask=attention_mask, loss_mask=empty_mask)

    assert out["total_loss"] is not None
    assert out["total_loss"].item() == 0.0


# Deep supervision

def test_deep_supervision_one_loss_per_outer_step():
    cfg = dict(TINY)
    cfg["outer_steps_t"] = 3
    torch.manual_seed(0)
    m = TRMModel(**cfg)

    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()
    out = m(token_ids, puzzle_idx,
            attention_mask=attention_mask, loss_mask=loss_mask)

    assert len(out["step_losses"]) == 3
    assert len(out["all_logits"]) == 3
    assert out["final_logits"] is out["all_logits"][-1]


# Overfit a single fixed batch

@pytest.mark.slow
def test_loss_decreases_over_50_steps():
    """Overfit a single batch; loss should drop noticeably. Marked slow."""
    cfg = dict(TINY)
    cfg["outer_steps_t"] = 2
    torch.manual_seed(0)
    m = TRMModel(**cfg)

    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        out = m(token_ids, puzzle_idx,
                attention_mask=attention_mask, loss_mask=loss_mask)
        loss = out["total_loss"]
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop enough: start={losses[0]:.3f} end={losses[-1]:.3f}"
    )