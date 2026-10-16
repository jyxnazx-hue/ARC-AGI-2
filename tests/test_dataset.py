# tests/test_dataset.py
"""
Component 3 benchmark: dataset shapes, masking, and collate.
Run: python -m pytest tests/test_dataset.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pytest
# pyrefly: ignore[missing-import]
import torch

from src.core import dataset as ds
from src.core import tokenizer as tk
from src.core import augment as ag


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("build", "holdout", "eval_public", "synthetic")


# ---------------------------------------------------------------------------
# Synthetic task (no data needed)
# ---------------------------------------------------------------------------

SYNTHETIC_TASK = {
    "train": [
        {"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]},
        {"input": [[1, 1, 1], [2, 2, 2]], "output": [[2, 2, 2], [1, 1, 1]]},
    ],
    "test": [
        {"input": [[6, 7, 8], [9, 0, 1]], "output": [[1, 0, 9], [8, 7, 6]]},
    ],
}


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> list[Path]:
    p = tmp_path / "synthetic_task.json"
    p.write_text(json.dumps(SYNTHETIC_TASK))
    return [p]


# ---------------------------------------------------------------------------
# Basic invariants on the synthetic task
# ---------------------------------------------------------------------------

def test_length_without_augmentation(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, augment=False)
    assert len(dset) == 1


def test_length_with_augmentation(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    for n_cp in (0, 1, 2):
        dset = ds.ARCTrainDataset(
            synthetic_paths, pidx, n_color_perms=n_cp, augment=True
        )
        assert len(dset) == 8 * (1 + n_cp)


def test_sample_shapes(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx)
    s = dset[0]

    assert s["input_canvases"].shape == (5, 900)
    assert s["output_canvases"].shape == (5, 900)
    assert s["pair_mask"].shape == (5,)
    assert s["input_cell_mask"].shape == (5, 900)
    assert s["output_cell_mask"].shape == (5, 900)
    assert s["loss_mask"].shape == (5, 900)
    assert s["input_canvases"].dtype == torch.long
    assert s["pair_mask"].dtype == torch.long
    assert isinstance(s["puzzle_idx"], int)
    assert isinstance(s["variant_idx"], int)


def test_short_task_is_padded(synthetic_paths):
    """2 real train pairs + n_train_pairs=4 -> pair_mask [1,1,0,0,1]."""
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, augment=False)
    s = dset[0]
    assert s["pair_mask"].tolist() == [1, 1, 0, 0, 1]


def test_padded_slots_are_all_pad(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, augment=False)
    s = dset[0]
    # Slots 2 and 3 are the padded ones
    assert (s["input_canvases"][2] == tk.PAD).all()
    assert (s["input_canvases"][3] == tk.PAD).all()
    assert (s["output_canvases"][2] == tk.PAD).all()
    assert (s["output_canvases"][3] == tk.PAD).all()
    # Masks must be all 0 there
    assert (s["input_cell_mask"][2] == 0).all()
    assert (s["output_cell_mask"][2] == 0).all()
    assert (s["loss_mask"][2] == 0).all()


def test_loss_mask_only_at_test_output(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, augment=False)
    s = dset[0]
    for i in range(4):
        assert (s["loss_mask"][i] == 0).all(), f"loss_mask row {i} should be zero"
    # Test row: should equal output_cell_mask at row 4
    assert torch.equal(s["loss_mask"][4], s["output_cell_mask"][4])
    assert s["loss_mask"][4].sum() == 6   # 3x2 real output grid


def test_cell_masks_match_canvas_pads(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, augment=False)
    s = dset[0]
    for i in range(5):
        in_non_pad = (s["input_canvases"][i] != tk.PAD)
        out_non_pad = (s["output_canvases"][i] != tk.PAD)
        assert torch.equal(s["input_cell_mask"][i].bool(), in_non_pad)
        assert torch.equal(s["output_cell_mask"][i].bool(), out_non_pad)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def test_collate_shapes(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx)
    batch = ds.collate_fn([dset[0], dset[1], dset[2]])

    assert batch["puzzle_idx"].shape == (3,)
    assert batch["input_canvases"].shape == (3, 5, 900)
    assert batch["output_canvases"].shape == (3, 5, 900)
    assert batch["pair_mask"].shape == (3, 5)
    assert batch["input_cell_mask"].shape == (3, 5, 900)
    assert batch["output_cell_mask"].shape == (3, 5, 900)
    assert batch["loss_mask"].shape == (3, 5, 900)
    assert isinstance(batch["task_id"], list)
    assert len(batch["task_id"]) == 3


def test_build_sequence_shapes(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx)
    batch = ds.collate_fn([dset[0], dset[1]])

    token_ids, attn, loss = ds.build_sequence(batch, puzzle_len=16)
    expected_T = 16 + 5 * 2 * 900
    assert token_ids.shape == (2, expected_T)
    assert attn.shape == (2, expected_T)
    assert loss.shape == (2, expected_T)
    # Puzzle positions: all attendable, no loss
    assert attn[:, :16].all()
    assert not loss[:, :16].any()


# ---------------------------------------------------------------------------
# puzzle_idx consistency
# ---------------------------------------------------------------------------

def test_puzzle_idx_stable_across_variants(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, n_color_perms=2)
    seen = set()
    for i in range(len(dset)):
        seen.add(dset[i]["puzzle_idx"])
    assert seen == {0}


# ---------------------------------------------------------------------------
# Augmentation changes the canvas
# ---------------------------------------------------------------------------

def test_augmented_variant_differs_from_identity(synthetic_paths):
    pidx = ds.build_puzzle_idx_map(synthetic_paths)
    dset = ds.ARCTrainDataset(synthetic_paths, pidx, n_color_perms=0)
    # variant 0 = identity, variant 1 = rotate 90 (asymmetric grid changes)
    a = dset[0]["input_canvases"][0]
    b = dset[1]["input_canvases"][0]
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------

def _first_tasks(n: int) -> list[Path]:
    for split in SPLITS:
        d = DATA_ROOT / split
        if d.exists():
            paths = sorted(d.glob("*.json"))[:n]
            if paths:
                return paths
    return []


def test_real_data_batch():
    paths = _first_tasks(4)
    if not paths:
        pytest.skip("no tasks available")
    pidx = ds.build_puzzle_idx_map(paths)
    dset = ds.ARCTrainDataset(paths, pidx, n_color_perms=2, augment=True)
    assert len(dset) == 4 * 8 * (1 + 2)

    batch = ds.collate_fn([dset[i] for i in range(4)])
    token_ids, attn, loss = ds.build_sequence(batch)

    assert token_ids.shape[0] == 4
    assert token_ids.shape[1] == 16 + 10 * 900
    # Loss only at test-output positions: sum equals 4 * (cells in test outputs)
    assert loss.sum().item() > 0
    # Puzzle positions never in loss
    assert not loss[:, :16].any()