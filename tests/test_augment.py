from __future__ import annotations

import json
from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pytest

from src.core import tokenizer as tk
from src.core import augment as ag
from src.core.grid import Grid


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("build", "holdout", "eval_public", "synthetic")


def _list_tasks(split: str, limit: int) -> list[Path]:
    d = DATA_ROOT / split
    return sorted(d.glob("*.json"))[:limit] if d.exists() else []


def _first_tasks(n: int) -> list[Path]:
    for split in SPLITS:
        paths = _list_tasks(split, limit=n)
        if paths:
            return paths
    return []


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


# Shape / count invariants

def test_canvas_count_and_shapes_single_task():
    paths = _first_tasks(1)
    if not paths:
        pytest.skip("no tasks available")
    task = _load(paths[0])
    out = ag.task_to_canvases(task, task_id=paths[0].stem)

    n_train = len(task["train"])
    n_test = len(task["test"])

    assert out["task_id"] == paths[0].stem
    assert out["train_inputs"].shape == (n_train, 30, 30)
    assert out["train_outputs"].shape == (n_train, 30, 30)
    assert out["test_inputs"].shape == (n_test, 30, 30)
    assert out["test_outputs"].shape == (n_test, 30, 30)
    assert out["test_outputs_masked"].shape == (n_test, 30, 30)
    assert len(out["test_output_shapes"]) == n_test


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_canvas_count_and_shapes_five_tasks(idx):
    paths = _first_tasks(5)
    if idx >= len(paths):
        pytest.skip("not enough tasks")
    task = _load(paths[idx])
    out = ag.task_to_canvases(task)

    assert out["train_inputs"].shape[0] == len(task["train"])
    assert out["test_inputs"].shape[0] == len(task["test"])

    for key in ("train_inputs", "train_outputs", "test_inputs",
                "test_outputs", "test_outputs_masked"):
        arr = out[key]
        assert arr.dtype == np.int64
        assert arr.ndim == 3
        assert arr.shape[1:] == (30, 30)

def test_every_grid_roundtrips():
    paths = _first_tasks(3)
    if not paths:
        pytest.skip("no tasks available")
    for p in paths:
        task = _load(p)
        out = ag.task_to_canvases(task)
        for i, pair in enumerate(task["train"]):
            g = np.asarray(pair["input"], dtype=np.int64)
            assert np.array_equal(tk.decode_canvas(out["train_inputs"][i]), g), \
                f"{p.stem} train[{i}].input"
            g = np.asarray(pair["output"], dtype=np.int64)
            assert np.array_equal(tk.decode_canvas(out["train_outputs"][i]), g), \
                f"{p.stem} train[{i}].output"


# Masked test output

def test_masked_test_output_fills_ground_truth_region():
    paths = _first_tasks(3)
    if not paths:
        pytest.skip("no tasks available")
    for p in paths:
        task = _load(p)
        out = ag.task_to_canvases(task)
        for i, pair in enumerate(task["test"]):
            if "output" not in pair:
                continue
            h, w = out["test_output_shapes"][i]
            masked = out["test_outputs_masked"][i]
            assert (masked[:h, :w] == tk.MASK).all(), f"{p.stem} test[{i}] mask region"
            assert (masked[h:, :] == tk.PAD).all(), f"{p.stem} test[{i}] below mask"
            assert (masked[:, w:] == tk.PAD).all(), f"{p.stem} test[{i}] right of mask"


# Augmentation

def test_augment_returns_eight_d4_variants():
    paths = _first_tasks(1)
    if not paths:
        pytest.skip("no tasks available")
    task = _load(paths[0])
    variants = ag.augment_task(task, n_color_perms=0)
    assert len(variants) == 8
    for v in variants:
        assert v["train_inputs"].shape == (len(task["train"]), 30, 30)
        assert v["test_inputs"].shape == (len(task["test"]), 30, 30)


def test_augment_with_color_perms_scales():
    paths = _first_tasks(1)
    if not paths:
        pytest.skip("no tasks available")
    task = _load(paths[0])
    variants = ag.augment_task(task, n_color_perms=2, seed=0)
    assert len(variants) == 8 * (1 + 2)


def test_d4_variants_preserve_pair_count():
    paths = _first_tasks(1)
    if not paths:
        pytest.skip("no tasks available")
    task = _load(paths[0])
    for v in ag.augment_task(task, n_color_perms=0):
        assert v["train_inputs"].shape[0] == len(task["train"])
        assert v["test_inputs"].shape[0] == len(task["test"])


def test_color_permutation_is_bijection_on_used_colors():
    paths = _first_tasks(1)
    if not paths:
        pytest.skip("no tasks available")
    task = _load(paths[0])
    grids = [Grid(p["input"]) for p in task["train"]] + \
            [Grid(p["output"]) for p in task["train"]]
    rng = np.random.default_rng(0)
    mapping = ag.random_color_permutation(grids, rng)

    used = sorted({int(c) for g in grids for c in np.unique(g.matrix)})
    assert sorted(mapping.keys()) == used
    assert sorted(mapping.values()) == used


def test_color_permutation_swap_does_not_collide():
    """The classic {1:2, 2:1} swap must not lose information."""
    g = Grid(np.array([[1, 2, 1], [2, 1, 2]], dtype=np.int8))
    swapped = ag.apply_color_permutation(g, {1: 2, 2: 1})
    expected = np.array([[2, 1, 2], [1, 2, 1]], dtype=np.int8)
    assert np.array_equal(swapped.matrix, expected)


def test_color_permutation_leaves_unmapped_colors_alone():
    g = Grid(np.array([[1, 3, 5]], dtype=np.int8))
    out = ag.apply_color_permutation(g, {1: 7})  
    assert out.matrix.tolist() == [[7, 3, 5]]