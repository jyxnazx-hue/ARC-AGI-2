from __future__ import annotations

# pyrefly: ignore[missing-import]
import torch 
# pyrefly: ignore[missing-import]
import pytest

from src.models.trm_core import TRMModel


TINY = dict(
    vocab_size=12, d_model=32, n_heads=2, n_layers=1, ffn_dim=64,
    inner_steps_k=2, outer_steps_t=8,  
    n_puzzles=4, puzzle_len=2, n_pairs=2,
    canvas_h=4, canvas_w=4,
)
TINY_T = TINY["puzzle_len"] + TINY["n_pairs"] * 2 * (TINY["canvas_h"] * TINY["canvas_w"])
B = 2


def _batch():
    torch.manual_seed(0)
    token_ids = torch.randint(0, 12, (B, TINY_T))
    puzzle_idx = torch.zeros(B, dtype=torch.long)
    attention_mask = torch.ones(B, TINY_T, dtype=torch.bool)
    loss_mask = torch.zeros(B, TINY_T, dtype=torch.bool)
    loss_mask[:, -16:] = True
    return token_ids, puzzle_idx, attention_mask, loss_mask


def test_tbptt_window_4_runs():
    """outer_steps_t=8, tbptt_window=4 -> detach at step 4. Should not error."""
    torch.manual_seed(0)
    m = TRMModel(**TINY)
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    out = m(token_ids, puzzle_idx,
            attention_mask=attention_mask, loss_mask=loss_mask,
            tbptt_window=4)
    assert len(out["step_losses"]) == 8
    assert torch.isfinite(out["total_loss"])


def test_tbptt_window_larger_than_outer_steps_runs():
    """tbptt_window > outer_steps_t -> never detach. Still works."""
    torch.manual_seed(0)
    m = TRMModel(**TINY)
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    out = m(token_ids, puzzle_idx,
            attention_mask=attention_mask, loss_mask=loss_mask,
            tbptt_window=100)
    assert torch.isfinite(out["total_loss"])


def test_backward_produces_grads_on_all_params():
    """Every trainable parameter must receive a gradient."""
    torch.manual_seed(0)
    m = TRMModel(**TINY)
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    out = m(token_ids, puzzle_idx,
            attention_mask=attention_mask, loss_mask=loss_mask,
            tbptt_window=4)
    out["total_loss"].backward()

    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    assert missing == [], f"no grad on: {missing}"


def test_grads_are_finite():
    torch.manual_seed(0)
    m = TRMModel(**TINY)
    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()

    out = m(token_ids, puzzle_idx,
            attention_mask=attention_mask, loss_mask=loss_mask,
            tbptt_window=4)
    out["total_loss"].backward()

    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"


def test_two_consecutive_backwards_do_not_explode():
    """Simulate two training steps; second loss/grad must stay finite."""
    torch.manual_seed(0)
    m = TRMModel(**TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    token_ids, puzzle_idx, attention_mask, loss_mask = _batch()
    for _ in range(2):
        opt.zero_grad()
        out = m(token_ids, puzzle_idx,
                attention_mask=attention_mask, loss_mask=loss_mask,
                tbptt_window=4)
        out["total_loss"].backward()
        opt.step()
        assert torch.isfinite(out["total_loss"])