from __future__ import annotations

import json
from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pytest

from src.core import tokenizer as tk

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("build", "holdout", "eval_public", "synthetic")

NAMED_TASK = "0a1d4ef5"


def _find_task_path(task_id: str) -> Path | None:
    for split in SPLITS:
        p = DATA_ROOT / split / f"{task_id}.json"
        if p.exists():
            return p
    # fallback: recursive search
    matches = list(DATA_ROOT.glob(f"**/{task_id}.json"))
    return matches[0] if matches else None


def _load_task(task_id: str) -> dict:
    path = _find_task_path(task_id)
    if path is None:
        pytest.skip(f"task {task_id} not found under {DATA_ROOT}")
    return json.loads(path.read_text())


def _list_tasks(limit: int = 5) -> list[str]:
    for split in SPLITS:
        d = DATA_ROOT / split
        if d.exists():
            ids = sorted(p.stem for p in d.glob("*.json"))
            if ids:
                return ids[:limit]
    return []

def test_vocab_size():
    assert tk.VOCAB_SIZE == 12
    assert tk.PAD == 0
    assert tk.MASK == 1
    assert tk.COLOR_OFFSET == 2
    assert len(tk.COLOR_TOKEN_IDS) == 10
    assert tk.COLOR_TOKEN_IDS[0] == 2
    assert tk.COLOR_TOKEN_IDS[-1] == 11


def test_canvas_constants():
    assert tk.CANVAS_H == 30
    assert tk.CANVAS_W == 30
    assert tk.CANVAS_SIZE == 900


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (3, 3), (5, 20), (20, 5), (30, 30), (7, 13)],
)
def test_encode_shape_and_pad_count(shape):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 10, size=shape, dtype=np.int64)
    canvas = tk.encode_grid(g)

    assert canvas.shape == (30, 30)
    assert canvas.dtype == np.int64
    assert int((canvas == tk.PAD).sum()) == 900 - g.size

    h, w = shape
    assert np.array_equal(canvas[:h, :w], g + tk.COLOR_OFFSET)
    # Everything outside is PAD
    assert np.all(canvas[h:, :] == tk.PAD)
    assert np.all(canvas[:, w:] == tk.PAD)


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (3, 3), (5, 20), (20, 5), (30, 30), (7, 13)],
)
def test_roundtrip_random(shape):
    rng = np.random.default_rng(1)
    g = rng.integers(0, 10, size=shape, dtype=np.int64)
    assert np.array_equal(tk.decode_canvas(tk.encode_grid(g)), g)


def test_roundtrip_all_black():
    g = np.zeros((4, 4), dtype=np.int64)
    assert np.array_equal(tk.decode_canvas(tk.encode_grid(g)), g)


def test_encode_oversized_raises():
    with pytest.raises(ValueError):
        tk.encode_grid(np.zeros((31, 5), dtype=np.int64))
    with pytest.raises(ValueError):
        tk.encode_grid(np.zeros((5, 31), dtype=np.int64))


def test_encode_bad_color_raises():
    with pytest.raises(ValueError):
        tk.encode_grid(np.array([[10]], dtype=np.int64))
    with pytest.raises(ValueError):
        tk.encode_grid(np.array([[-1]], dtype=np.int64))


def test_decode_masked_raises():
    c = tk.encode_grid(np.zeros((4, 4), dtype=np.int64))
    c[0, 0] = tk.MASK
    with pytest.raises(ValueError):
        tk.decode_canvas(c)


def test_decode_all_pad_returns_empty():
    c = np.full((30, 30), tk.PAD, dtype=np.int64)
    out = tk.decode_canvas(c)
    assert out.shape == (0, 0)


def test_flatten_unflatten_roundtrip():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 10, size=(6, 7), dtype=np.int64)
    canvas = tk.encode_grid(g)
    flat = tk.flatten(canvas)
    assert flat.shape == (900,)
    assert np.array_equal(tk.unflatten(flat), canvas)


def test_argmax_to_colors_never_picks_pad_or_mask():
    logits = np.zeros((10, 12))
    logits[:, tk.PAD] = 100.0
    logits[:, tk.MASK] = 100.0
    picks = tk.argmax_to_colors(logits)
    assert np.all((picks >= 2) & (picks <= 11))


def test_color_logit_mask():
    m = tk.color_logit_mask()
    assert m.shape == (12,)
    assert m[0] is np.False_ or not m[0]
    assert m[1] is np.False_ or not m[1]
    assert all(m[i] for i in range(2, 12))

# Data-backed benchmark: 5 tasks

def _check_task_grids(task: dict) -> int:
    checked = 0
    for pair in task["train"]:
        for key in ("input", "output"):
            g = np.asarray(pair[key], dtype=np.int64)
            assert np.array_equal(tk.decode_canvas(tk.encode_grid(g)), g), (
                f"round-trip failed on {key} shape {g.shape}"
            )
            checked += 1
    for pair in task["test"]:
        g = np.asarray(pair["input"], dtype=np.int64)
        assert np.array_equal(tk.decode_canvas(tk.encode_grid(g)), g)
        checked += 1
    return checked


def test_named_task_roundtrip():
    task = _load_task(NAMED_TASK)
    n = _check_task_grids(task)
    assert n > 0


def test_five_tasks_roundtrip():
    ids = _list_tasks(limit=5)
    if not ids:
        pytest.skip(f"no tasks found under {DATA_ROOT}")
    total = 0
    for tid in ids:
        total += _check_task_grids(_load_task(tid))
    assert total > 0