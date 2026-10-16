"""
Task serialization into fixed 30x30 canvases + D4 / color-perm augmentation.
  - `test_outputs`         : real ground truth (for loss at training time)
  - `test_outputs_masked`  : MASK fills the answer region (the model's
                             initial test-output state)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# pyrefly: ignore[missing-import]
import numpy as np

from src.core import tokenizer as tk
from src.core.grid import Grid



def load_task(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

# Grid -> canvas helpers

def _as_grid(x) -> Grid:
    return x if isinstance(x, Grid) else Grid(x)


def grid_to_canvas(grid) -> np.ndarray:
    """Grid / nested list -> (30, 30) int64 canvas, placed top-left."""
    g = _as_grid(grid)
    return tk.encode_grid(g.matrix)


def masked_canvas_for(grid) -> np.ndarray:
    """
    (30, 30) canvas where the answer region is MASK and everywhere else PAD.
    Used as the model's initial test-output state during training.
    """
    g = _as_grid(grid)
    h, w = g.shape
    if h > tk.CANVAS_H or w > tk.CANVAS_W:
        raise ValueError(f"grid {h}x{w} exceeds canvas")
    canvas = np.full((tk.CANVAS_H, tk.CANVAS_W), tk.PAD, dtype=np.int64)
    canvas[:h, :w] = tk.MASK
    return canvas

# task_to_canvases

def task_to_canvases(
    task: Dict[str, Any], task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert one ARC task into fixed-canvas numpy arrays.

    Returns a dict:
        task_id              : str or None
        train_inputs         : (n_train, 30, 30) int64
        train_outputs        : (n_train, 30, 30) int64
        test_inputs          : (n_test,  30, 30) int64
        test_outputs         : (n_test,  30, 30) int64  (ground truth, or PAD if absent)
        test_outputs_masked  : (n_test,  30, 30) int64  (MASK in answer region)
        test_output_shapes   : list of (H, W)
    """
    if "train" not in task or "test" not in task:
        raise ValueError("task must have 'train' and 'test' keys")
    if len(task["train"]) == 0:
        raise ValueError("task has no training pairs")

    train_inputs, train_outputs = [], []
    for i, pair in enumerate(task["train"]):
        for k in ("input", "output"):
            if k not in pair:
                raise ValueError(f"train pair {i} missing '{k}'")
        train_inputs.append(grid_to_canvas(pair["input"]))
        train_outputs.append(grid_to_canvas(pair["output"]))

    test_inputs, test_outputs, test_outputs_masked = [], [], []
    test_output_shapes: List[Tuple[int, int]] = []
    for i, pair in enumerate(task["test"]):
        if "input" not in pair:
            raise ValueError(f"test pair {i} missing 'input'")
        test_inputs.append(grid_to_canvas(pair["input"]))

        if "output" in pair:
            out_grid = _as_grid(pair["output"])
            test_outputs.append(grid_to_canvas(pair["output"]))
            test_outputs_masked.append(masked_canvas_for(pair["output"]))
            test_output_shapes.append(tuple(out_grid.shape))
        else:
            in_shape = _as_grid(pair["input"]).shape
            test_outputs.append(
                np.full((tk.CANVAS_H, tk.CANVAS_W), tk.PAD, dtype=np.int64)
            )
            test_outputs_masked.append(masked_canvas_for(pair["input"]))
            test_output_shapes.append(tuple(in_shape))

    return {
        "task_id": task_id,
        "train_inputs": np.stack(train_inputs, axis=0),
        "train_outputs": np.stack(train_outputs, axis=0),
        "test_inputs": np.stack(test_inputs, axis=0),
        "test_outputs": np.stack(test_outputs, axis=0),
        "test_outputs_masked": np.stack(test_outputs_masked, axis=0),
        "test_output_shapes": test_output_shapes,
    }


# Augmentation: D4 + color permutation

def _d4_transform(k: int):
    """k in 0..7: 0-3 rotations, 4-7 rotations followed by horizontal flip."""
    def tf(g: Grid) -> Grid:
        out = g.rotate(k=k % 4)
        if k >= 4:
            out = out.flip_horizontal()
        return out
    return tf


def random_color_permutation(
    grids: Sequence[Grid], rng: np.random.Generator
) -> Dict[int, int]:
    """Random bijection over the colors that appear in `grids`."""
    used = sorted({int(c) for g in grids for c in np.unique(g.matrix)})
    if not used:
        return {}
    shuffled = [int(x) for x in rng.permutation(used)]
    return {int(o): int(n) for o, n in zip(used, shuffled)}


def apply_color_permutation(grid: Grid, mapping: Dict[int, int]) -> Grid:
    """
    Apply a color permutation in one pass. Avoids the swap-collision bug in
    `Grid.recolor` (e.g. {1: 2, 2: 1} would otherwise lose information).
    """
    lut = np.arange(10, dtype=np.int8)
    for old, new in mapping.items():
        lut[old] = new
    return Grid(lut[grid.matrix])


def _transform_task(task: Dict[str, Any], transform) -> Dict[str, Any]:
    """Apply a Grid -> Grid transform consistently across the whole task."""
    new_task: Dict[str, Any] = {"train": [], "test": []}
    for pair in task["train"]:
        new_task["train"].append({
            "input": transform(Grid(pair["input"])).to_list(),
            "output": transform(Grid(pair["output"])).to_list(),
        })
    for pair in task["test"]:
        new_pair = {"input": transform(Grid(pair["input"])).to_list()}
        if "output" in pair:
            new_pair["output"] = transform(Grid(pair["output"])).to_list()
        new_task["test"].append(new_pair)
    return new_task


def augment_task(
    task: Dict[str, Any],
    task_id: Optional[str] = None,
    n_color_perms: int = 0,
    seed: Optional[int] = None,
    include_identity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Produce augmented variants of a task, each in the same format as
    `task_to_canvases`.

    Variants:
      * 8 D4 symmetries (identity + 3 rotations + 4 flipped rotations)
      * optionally, `n_color_perms` random color permutations stacked on top
        of each D4 view, sampled over the colors used in the whole task.

    Total variants = 8 * (1 + n_color_perms).
    """
    rng = np.random.default_rng(seed)

    all_grids: List[Grid] = []
    if n_color_perms > 0:
        for pair in task["train"]:
            all_grids.append(Grid(pair["input"]))
            all_grids.append(Grid(pair["output"]))
        for pair in task["test"]:
            all_grids.append(Grid(pair["input"]))
            if "output" in pair:
                all_grids.append(Grid(pair["output"]))

    variants: List[Dict[str, Any]] = []
    for k in range(8):
        if k == 0 and not include_identity:
            continue
        d4_fn = _d4_transform(k)

        variants.append(
            task_to_canvases(_transform_task(task, d4_fn), task_id=task_id)
        )

        for _ in range(n_color_perms):
            mapping = random_color_permutation(all_grids, rng)

            def combined(g, _fn=d4_fn, _m=mapping):
                return apply_color_permutation(_fn(g), _m)

            variants.append(
                task_to_canvases(
                    _transform_task(task, combined), task_id=task_id
                )
            )

    return variants

def _all_grids(task: Dict[str, Any]) -> List[Grid]:
    """Every grid in a task, as Grid objects."""
    grids: List[Grid] = []
    for pair in task["train"]:
        grids.append(Grid(pair["input"]))
        grids.append(Grid(pair["output"]))
    for pair in task["test"]:
        grids.append(Grid(pair["input"]))
        if "output" in pair:
            grids.append(Grid(pair["output"]))
    return grids


def augment_task_variant(
    task: Dict[str, Any],
    variant_idx: int,
    n_color_perms: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Apply ONE augmentation variant to `task`, identified by `variant_idx`
    in `[0, 8 * (1 + n_color_perms))`. Cheaper than `augment_task` when you
    only need one variant (which is what the dataset does).

    Ordering matches `augment_task`:
      variant_idx // (1 + n_color_perms) in 0..7  -> D4 index
      variant_idx %  (1 + n_color_perms) in 0..n_color_perms
          0 = no color perm, else k-th color perm
    """
    n_per_d4 = 1 + n_color_perms
    if not 0 <= variant_idx < 8 * n_per_d4:
        raise ValueError(
            f"variant_idx {variant_idx} out of range [0, {8 * n_per_d4})"
        )

    d4_k = (variant_idx // n_per_d4) % 8
    color_idx = variant_idx % n_per_d4

    d4_fn = _d4_transform(d4_k)
    task_d4 = _transform_task(task, d4_fn)

    if color_idx == 0:
        return task_d4

    rng = np.random.default_rng(seed + 7919 * variant_idx)
    mapping = random_color_permutation(_all_grids(task), rng)
    return _transform_task(
        task_d4, lambda g: apply_color_permutation(g, mapping)
    )