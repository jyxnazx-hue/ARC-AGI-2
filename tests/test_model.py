from __future__ import annotations

import inspect
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import pytest

from src.models.trm_core import TRMModel

TINY = dict(
    vocab_size=12, d_model=32, n_heads=2, n_layers=1, ffn_dim=64,
    inner_steps_k=2, outer_steps_t=2,
    n_puzzles=16, puzzle_len=2, n_pairs=2,
    canvas_h=4, canvas_w=4,
)
TINY_T = TINY["puzzle_len"] + TINY["n_pairs"] * 2 * (TINY["canvas_h"] * TINY["canvas_w"])
TINY_B = 2


@pytest.fixture
def tiny_model() -> TRMModel:
    torch.manual_seed(0)
    return TRMModel(**TINY)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(0)
    token_ids = torch.randint(0, 12, (TINY_B, TINY_T))
    puzzle_idx = torch.randint(0, TINY["n_puzzles"], (TINY_B,))
    attention_mask = torch.ones(TINY_B, TINY_T, dtype=torch.bool)
    loss_mask = torch.zeros(TINY_B, TINY_T, dtype=torch.bool)
    loss_mask[:, -TINY["canvas_h"] * TINY["canvas_w"]:] = True
    return token_ids, puzzle_idx, attention_mask, loss_mask


# Param count

def test_default_puzzle_len_is_8():
    sig = inspect.signature(TRMModel.__init__)
    assert sig.parameters["puzzle_len"].default == 8


def test_param_count_fits_budget():
    m = TRMModel(
        vocab_size=12, d_model=512, n_heads=8, n_layers=2, ffn_dim=1024,
        n_puzzles=800, puzzle_len=8, n_pairs=5,
    )
    n = m.count_parameters()
    puzzle = m.puzzle_embeddings.weight.numel()
    core = n - puzzle

    assert 8_000_000 < n < 9_500_000, f"total params: {n}"
    assert 4_800_000 < core < 6_000_000, f"core params: {core}"
    assert puzzle == 801 * 8 * 512


# Shapes

def test_forward_shapes(tiny_model, tiny_batch):
    token_ids, puzzle_idx, attention_mask, loss_mask = tiny_batch
    out = tiny_model(token_ids, puzzle_idx,
                     attention_mask=attention_mask, loss_mask=loss_mask)
    assert out["final_logits"].shape == (TINY_B, TINY_T, 12)
    for lg in out["all_logits"]:
        assert lg.shape == (TINY_B, TINY_T, 12)
    assert out["total_loss"] is not None
    assert out["total_loss"].ndim == 0


def test_forward_no_loss(tiny_model, tiny_batch):
    token_ids, puzzle_idx, attention_mask, _ = tiny_batch
    out = tiny_model(token_ids, puzzle_idx, attention_mask=attention_mask)
    assert out["total_loss"] is None
    assert len(out["step_losses"]) == 0
    assert out["final_logits"].shape == (TINY_B, TINY_T, 12)


def test_forward_wrong_seq_len_raises(tiny_model):
    token_ids = torch.zeros(TINY_B, TINY_T - 1, dtype=torch.long)
    puzzle_idx = torch.zeros(TINY_B, dtype=torch.long)
    with pytest.raises(ValueError, match="expected T"):
        tiny_model(token_ids, puzzle_idx)


# 2D positional encoding

def test_row_col_embeddings_exist(tiny_model):
    assert tiny_model.row_embeddings.weight.shape == (TINY["canvas_h"], TINY["d_model"])
    assert tiny_model.col_embeddings.weight.shape == (TINY["canvas_w"], TINY["d_model"])


def test_2d_pos_encoding_differs_for_same_row(tiny_model):
    row_emb = tiny_model.row_embeddings.weight[1]
    col_0 = tiny_model.col_embeddings.weight[0]
    col_1 = tiny_model.col_embeddings.weight[1]
    assert not torch.allclose(row_emb + col_0, row_emb + col_1)


# Puzzle embedding

def test_null_puzzle_embedding_is_zero(tiny_model):
    null = tiny_model.puzzle_embeddings.weight[tiny_model.n_puzzles]
    assert torch.all(null == 0)


def test_puzzle_idx_changes_output(tiny_model, tiny_batch):
    token_ids, _, attention_mask, loss_mask = tiny_batch
    p_a = torch.zeros(TINY_B, dtype=torch.long)
    p_b = torch.ones(TINY_B, dtype=torch.long)

    torch.manual_seed(42)
    out_a = tiny_model(token_ids, p_a, attention_mask=attention_mask,
                       loss_mask=loss_mask)
    torch.manual_seed(42)
    out_b = tiny_model(token_ids, p_b, attention_mask=attention_mask,
                       loss_mask=loss_mask)

    diff = (out_a["final_logits"] - out_b["final_logits"]).abs().mean()
    assert diff.item() > 1e-4


def test_null_puzzle_idx_runs(tiny_model, tiny_batch):
    token_ids, _, attention_mask, loss_mask = tiny_batch
    null_idx = torch.full((TINY_B,), tiny_model.n_puzzles, dtype=torch.long)
    out = tiny_model(token_ids, null_idx,
                     attention_mask=attention_mask, loss_mask=loss_mask)
    assert out["final_logits"].shape == (TINY_B, TINY_T, 12)


def test_out_of_range_puzzle_idx_is_clamped(tiny_model, tiny_batch):
    token_ids, _, attention_mask, loss_mask = tiny_batch
    big = torch.full((TINY_B,), 10_000, dtype=torch.long)
    out = tiny_model(token_ids, big,
                     attention_mask=attention_mask, loss_mask=loss_mask)
    assert out["final_logits"].shape == (TINY_B, TINY_T, 12)


# Bidirectional attention

def test_attention_is_bidirectional(tiny_model, tiny_batch):
    token_ids, puzzle_idx, attention_mask, loss_mask = tiny_batch
    token_ids_a = token_ids.clone()
    token_ids_b = token_ids.clone()
    token_ids_b[:, -1] = (token_ids_b[:, -1] + 1) % 12

    torch.manual_seed(7)
    out_a = tiny_model(token_ids_a, puzzle_idx,
                       attention_mask=attention_mask, loss_mask=loss_mask)
    torch.manual_seed(7)
    out_b = tiny_model(token_ids_b, puzzle_idx,
                       attention_mask=attention_mask, loss_mask=loss_mask)

    diff_pos0 = (out_a["final_logits"][:, 0] -
                 out_b["final_logits"][:, 0]).abs().mean()
    assert diff_pos0.item() > 0


# Loss plumbing

def test_loss_is_finite(tiny_model, tiny_batch):
    token_ids, puzzle_idx, attention_mask, loss_mask = tiny_batch
    out = tiny_model(token_ids, puzzle_idx,
                     attention_mask=attention_mask, loss_mask=loss_mask)
    assert torch.isfinite(out["total_loss"])


def test_loss_sensitive_to_masked_positions(tiny_model, tiny_batch):
    """Loss should change more when we perturb a masked (loss) position."""
    token_ids, puzzle_idx, attention_mask, loss_mask = tiny_batch

    torch.manual_seed(3)
    base = tiny_model(token_ids, puzzle_idx,
                      attention_mask=attention_mask, loss_mask=loss_mask)

    ids_out = token_ids.clone()
    ids_out[:, 0] = (ids_out[:, 0] + 1) % 12   # puzzle position, no loss
    torch.manual_seed(3)
    out_outside = tiny_model(ids_out, puzzle_idx,
                             attention_mask=attention_mask, loss_mask=loss_mask)

    ids_in = token_ids.clone()
    ids_in[:, -1] = (ids_in[:, -1] + 1) % 12   # loss position
    torch.manual_seed(3)
    out_inside = tiny_model(ids_in, puzzle_idx,
                            attention_mask=attention_mask, loss_mask=loss_mask)

    d_outside = (base["total_loss"] - out_outside["total_loss"]).abs().item()
    d_inside = (base["total_loss"] - out_inside["total_loss"]).abs().item()
    assert d_inside > d_outside, (d_inside, d_outside)

@pytest.mark.slow
def test_real_size_forward_smoke():
    """Full-size forward, batch=1. Skipped unless -m slow."""
    torch.manual_seed(0)
    m = TRMModel(
        vocab_size=12, d_model=512, n_heads=8, n_layers=2, ffn_dim=1024,
        inner_steps_k=4, outer_steps_t=3,
        n_puzzles=800, puzzle_len=8, n_pairs=5,
    )
    T = m.expected_seq_len
    token_ids = torch.randint(0, 12, (1, T))
    puzzle_idx = torch.zeros(1, dtype=torch.long)
    attention_mask = torch.ones(1, T, dtype=torch.bool)
    loss_mask = torch.zeros(1, T, dtype=torch.bool)
    loss_mask[:, -900:] = True

    out = m(token_ids, puzzle_idx,
            attention_mask=attention_mask, loss_mask=loss_mask)
    assert out["final_logits"].shape == (1, T, 12)
    assert torch.isfinite(out["total_loss"])