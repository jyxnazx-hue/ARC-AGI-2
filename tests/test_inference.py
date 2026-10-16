from __future__ import annotations

import json
from pathlib import Path

import numpy as np # pyrefly: ignore[missing-import]
import pytest # pyrefly: ignore[missing-import]
import torch # pyrefly: ignore[missing-import]

from src.core import inference as inf
from src.core import tokenizer as tk
from train_trm import SMOKE, Config, build_model


FAST_TAU = [1.0, 0.5, 0.0]

# Shape inference

def test_shape_infer_preserves_when_all_pairs_preserve():
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[0, 0], [0, 0]], "output": [[0, 0], [0, 0]]},
        ],
        "test": [{"input": [[1, 2, 3], [4, 5, 6]]}],
    }
    assert inf.infer_answer_shape(task) == (2, 3)


def test_shape_infer_uses_uniform_output_shape():
    task = {
        "train": [
            {"input": [[0, 1, 2]], "output": [[1, 2]]},
            {"input": [[3, 4, 5, 6]], "output": [[5, 6]]},
        ],
        "test": [{"input": [[7, 8, 9]]}],
    }
    assert inf.infer_answer_shape(task) == (1, 2)


def test_shape_infer_uses_most_common():
    task = {
        "train": [
            {"input": [[1]], "output": [[1]]},
            {"input": [[1, 2]], "output": [[1, 2, 3]]},
            {"input": [[3]], "output": [[3]]},
        ],
        "test": [{"input": [[9]]}],
    }
    assert inf.infer_answer_shape(task) == (1, 1)

# Fixtures

@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Config:
    c = Config(**{**SMOKE.__dict__})
    c.ckpt_dir = tmp_path / "ckpt"
    return c


@pytest.fixture
def tiny_model(tiny_cfg: Config):
    torch.manual_seed(0)
    m = build_model(tiny_cfg).to(tiny_cfg.device)
    m.eval()
    return m


@pytest.fixture
def synthetic_task():
    return {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        ],
        "test": [{"input": [[5, 6], [7, 8]]}],
    }

# Single attempt

def test_single_attempt_returns_valid_grid(tiny_model, tiny_cfg, synthetic_task):
    grid = inf.predict_one_attempt(
        tiny_model, synthetic_task,
        puzzle_idx=tiny_model.n_puzzles,  # null embedding
        n_train_pairs=tiny_cfg.n_train_pairs,
        puzzle_len=tiny_cfg.puzzle_len,
        device=tiny_cfg.device,
        tau_schedule=FAST_TAU,
    )
    assert isinstance(grid, list)
    assert len(grid) == 2
    assert all(len(row) == 2 for row in grid)
    assert all(0 <= c <= 9 for row in grid for c in row)

#  Two attempts

def test_two_attempts_shapes_match(tiny_model, tiny_cfg, synthetic_task):
    a1, a2 = inf.predict_two_attempts(
        tiny_model, synthetic_task,
        puzzle_idx=tiny_model.n_puzzles,
        n_train_pairs=tiny_cfg.n_train_pairs,
        puzzle_len=tiny_cfg.puzzle_len,
        device=tiny_cfg.device,
        tau_schedule=FAST_TAU,
    )
    assert len(a1) == len(a2) == 2
    assert all(len(r) == 2 for r in a1)
    assert all(len(r) == 2 for r in a2)


def test_attempt_2_is_unrotated_correctly(tiny_model, tiny_cfg):
    """
    180-rotate a task, run attempt 2 logic manually, un-rotate, compare
    shape to original. This checks the un-rotate step, not the values.
    """
    task = {
        "train": [
            {"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]},
        ],
        "test": [{"input": [[6, 7, 8], [9, 0, 1]]}],
    }
    task_rot = inf._rot180_task(task)
    # 180 rotation should preserve shape
    assert np.asarray(task_rot["test"][0]["input"]).shape == (2, 3)
    back = inf._rot180_grid(task_rot["test"][0]["input"])
    assert back == task["test"][0]["input"]

# Submission writer

def test_write_submission_valid_json(tiny_model, tiny_cfg, tmp_path):
    tasks = {
        "task_a": {
            "train": [{"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]}],
            "test": [{"input": [[4, 5], [6, 7]]}],
        },
        "task_b": {
            "train": [{"input": [[1, 1]], "output": [[2, 2]]}],
            "test": [{"input": [[9, 9]]}],
        },
    }
    out = tmp_path / "sub.json"
    inf.write_submission(
        model=tiny_model,
        tasks=tasks,
        puzzle_idx_map=None,
        n_train_pairs=tiny_cfg.n_train_pairs,
        puzzle_len=tiny_cfg.puzzle_len,
        out_path=out,
        device=tiny_cfg.device,
        tau_schedule=FAST_TAU,
        verbose=False,
    )
    assert out.exists()

    data = json.loads(out.read_text())
    assert set(data.keys()) == {"task_a", "task_b"}
    for tid, entry in data.items():
        assert set(entry.keys()) == {"attempt_1", "attempt_2"}
        for key in ("attempt_1", "attempt_2"):
            grid = entry[key]
            assert isinstance(grid, list)
            assert len(grid) > 0
            assert all(isinstance(row, list) and len(row) > 0 for row in grid)


def test_real_task_attempt(tiny_model, tiny_cfg):
    """Run on one real task from data/build."""
    import json as _json
    build_dir = Path("data/build")
    paths = sorted(build_dir.glob("*.json"))[:1]
    if not paths:
        pytest.skip("no tasks in data/build")
    task = _json.loads(paths[0].read_text())

    a1, a2 = inf.predict_two_attempts(
        tiny_model, task,
        puzzle_idx=tiny_model.n_puzzles,
        n_train_pairs=tiny_cfg.n_train_pairs,
        puzzle_len=tiny_cfg.puzzle_len,
        device=tiny_cfg.device,
        tau_schedule=FAST_TAU,
    )
    expected_shape = inf.infer_answer_shape(task)
    assert np.asarray(a1).shape == expected_shape
    assert np.asarray(a2).shape == expected_shape