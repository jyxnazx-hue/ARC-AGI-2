# src/core/dataset.py
"""
ARC-AGI-2 training dataset (Sub-Step 1.2, Component 3).

Each sample represents ONE augmented variant of ONE task, laid out as
`n_train_pairs` (default 4) training pairs + 1 test pair:

    input_canvases   [5, 900]  — 4 train inputs + 1 test input
    output_canvases  [5, 900]  — 4 train outputs + 1 test output
    pair_mask        [5]       — 1 for a real pair, 0 for padded
    input_cell_mask  [5, 900]  — 1 for non-pad cells in the input canvas
    output_cell_mask [5, 900]  — 1 for non-pad cells in the output canvas
    loss_mask        [5, 900]  — 1 only where we compute loss (test output)

The final interleaved sequence fed to the model is
    [puzzle(16)] [in_0,out_0, in_1,out_1, ..., in_4,out_4]
of length 16 + 5 * 2 * 900 = 9016. Sequence assembly is done in the
training loop, not here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
from torch.utils.data import Dataset

from src.core import augment as ag
from src.core import tokenizer as tk

# puzzle_idx mapping

def build_puzzle_idx_map(task_paths: List[Path]) -> Dict[str, int]:
    """Deterministic task_id -> int mapping (sorted by id)."""
    return {p.stem: i for i, p in enumerate(sorted(task_paths))}


# Dataset

class ARCTrainDataset(Dataset):
    """
    Parameters
    ----------
    task_paths : list of Path
        One JSON file per task.
    puzzle_idx_map : dict[str, int]
        task_id -> puzzle embedding row. Must contain every task.
    n_train_pairs : int
        Fixed number of training pairs per sample. Default 4.
        Tasks with more pairs are randomly sub-sampled; tasks with fewer are
        zero-padded (pair_mask = 0).
    n_color_perms : int
        Number of random color permutations on top of each D4 view.
        Total variants per task = 8 * (1 + n_color_perms).
        Default 2 -> 24 variants / task (matches your doc).
    augment : bool
        If False, only the identity variant is used (1 row / task).
    seed : int
        Base RNG seed for pair sampling and color perms.
    """

    def __init__(
        self,
        task_paths: List[Path],
        puzzle_idx_map: Dict[str, int],
        n_train_pairs: int = 4,
        n_color_perms: int = 2,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.task_paths = [Path(p) for p in task_paths]
        self.puzzle_idx_map = puzzle_idx_map
        self.n_train_pairs = n_train_pairs
        self.n_total_pairs = n_train_pairs + 1  # + test
        self.n_color_perms = n_color_perms
        self.augment = augment
        self.seed = seed

        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.rows: List[Tuple[str, int, int]] = []  # (task_id, puzzle_idx, variant_idx)

        n_variants = 8 * (1 + n_color_perms) if augment else 1
        for path in self.task_paths:
            task_id = path.stem
            if task_id not in puzzle_idx_map:
                continue
            self.tasks[task_id] = json.loads(path.read_text())
            pidx = puzzle_idx_map[task_id]
            for v in range(n_variants):
                self.rows.append((task_id, pidx, v))

    def __len__(self) -> int:
        return len(self.rows)

    # internal helpers 

    def _variant_task(self, task_id: str, variant_idx: int) -> Dict[str, Any]:
        task = self.tasks[task_id]
        if not self.augment or variant_idx == 0:
            return task
        return ag.augment_task_variant(
            task,
            variant_idx=variant_idx,
            n_color_perms=self.n_color_perms,
            seed=self.seed + hash(task_id) % (2**31),
        )

    # Dataset API 

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_id, puzzle_idx, variant_idx = self.rows[idx]
        task = self._variant_task(task_id, variant_idx)

        rng = np.random.default_rng(self.seed + 104729 * idx)

        n_train = len(task["train"])
        if n_train >= self.n_train_pairs:
            selected = sorted(
                rng.choice(n_train, size=self.n_train_pairs, replace=False).tolist()
            )
        else:
            selected = list(range(n_train))

        n_pairs = self.n_total_pairs
        input_canvases = np.zeros((n_pairs, tk.CANVAS_SIZE), dtype=np.int64)
        output_canvases = np.zeros((n_pairs, tk.CANVAS_SIZE), dtype=np.int64)
        input_cell_mask = np.zeros((n_pairs, tk.CANVAS_SIZE), dtype=np.int64)
        output_cell_mask = np.zeros((n_pairs, tk.CANVAS_SIZE), dtype=np.int64)
        pair_mask = np.zeros(n_pairs, dtype=np.int64)

        # Training pairs
        for slot, pair_idx in enumerate(selected):
            pair = task["train"][pair_idx]
            in_canvas = ag.grid_to_canvas(pair["input"])
            out_canvas = ag.grid_to_canvas(pair["output"])
            input_canvases[slot] = tk.flatten(in_canvas)
            output_canvases[slot] = tk.flatten(out_canvas)
            input_cell_mask[slot] = tk.flatten(tk.pad_mask(in_canvas))
            output_cell_mask[slot] = tk.flatten(tk.pad_mask(out_canvas))
            pair_mask[slot] = 1

        # Test pair
        test_pair = task["test"][0]
        test_in = ag.grid_to_canvas(test_pair["input"])
        input_canvases[-1] = tk.flatten(test_in)
        input_cell_mask[-1] = tk.flatten(tk.pad_mask(test_in))
        pair_mask[-1] = 1

        if "output" in test_pair:
            test_out = ag.grid_to_canvas(test_pair["output"])
            output_canvases[-1] = tk.flatten(test_out)
            output_cell_mask[-1] = tk.flatten(tk.pad_mask(test_out))

        # Loss mask 
        loss_mask = np.zeros_like(output_cell_mask)
        loss_mask[-1] = output_cell_mask[-1]

        return {
            "task_id": task_id,
            "puzzle_idx": puzzle_idx,
            "variant_idx": variant_idx,
            "input_canvases": torch.from_numpy(input_canvases),      # [5, 900]
            "output_canvases": torch.from_numpy(output_canvases),    # [5, 900]
            "pair_mask": torch.from_numpy(pair_mask),                # [5]
            "input_cell_mask": torch.from_numpy(input_cell_mask),    # [5, 900]
            "output_cell_mask": torch.from_numpy(output_cell_mask),  # [5, 900]
            "loss_mask": torch.from_numpy(loss_mask),                # [5, 900]
        }


# Collate   

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack per-sample tensors along a new batch dimension."""
    return {
        "task_id": [b["task_id"] for b in batch],
        "puzzle_idx": torch.tensor(
            [b["puzzle_idx"] for b in batch], dtype=torch.long
        ),                                                          # [B]
        "variant_idx": torch.tensor(
            [b["variant_idx"] for b in batch], dtype=torch.long
        ),                                                          # [B]
        "input_canvases": torch.stack(
            [b["input_canvases"] for b in batch], dim=0
        ),                                                          # [B, 5, 900]
        "output_canvases": torch.stack(
            [b["output_canvases"] for b in batch], dim=0
        ),                                                          # [B, 5, 900]
        "pair_mask": torch.stack(
            [b["pair_mask"] for b in batch], dim=0
        ),                                                          # [B, 5]
        "input_cell_mask": torch.stack(
            [b["input_cell_mask"] for b in batch], dim=0
        ),                                                          # [B, 5, 900]
        "output_cell_mask": torch.stack(
            [b["output_cell_mask"] for b in batch], dim=0
        ),                                                          # [B, 5, 900]
        "loss_mask": torch.stack(
            [b["loss_mask"] for b in batch], dim=0
        ),                                                          # [B, 5, 900]
    }


# Sequence assembly 

def build_sequence(
    batch: Dict[str, Any],
    puzzle_len: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interleave input/output canvases into a single flat sequence per sample.

    Returns
    -------
    token_ids : [B, T] long, T = puzzle_len + n_pairs * 2 * 900
    attention : [B, T] bool, True where the model may attend
                (puzzle positions + non-pad cells of real pairs)
    loss      : [B, T] bool, True where cross-entropy is computed
    """
    B, n_pairs, _ = batch["input_canvases"].shape
    device = batch["input_canvases"].device

    # [B, n_pairs, 2, 900] -> [B, n_pairs*2*900]
    stacked = torch.stack(
        [batch["input_canvases"], batch["output_canvases"]], dim=2
    ).reshape(B, n_pairs * 2 * tk.CANVAS_SIZE)

    # Puzzle positions
    puzzle = torch.zeros(
        B, puzzle_len, dtype=torch.long, device=device
    )
    token_ids = torch.cat([puzzle, stacked], dim=1)

    # Attention mask
    pm = batch["pair_mask"].unsqueeze(-1)                      # [B, 5, 1]
    cm = torch.stack(
        [batch["input_cell_mask"], batch["output_cell_mask"]], dim=2
    ).reshape(B, n_pairs * 2 * tk.CANVAS_SIZE).bool()          # [B, 9000]
    pm_exp = pm.unsqueeze(2).expand(-1, -1, 2, tk.CANVAS_SIZE) \
              .reshape(B, n_pairs * 2 * tk.CANVAS_SIZE).bool() # [B, 9000]
    canvas_attn = pm_exp & cm

    puzzle_attn = torch.ones(B, puzzle_len, dtype=torch.bool, device=device)
    attention = torch.cat([puzzle_attn, canvas_attn], dim=1)

    # Loss mask
    lm = torch.stack(
        [
            torch.zeros_like(batch["input_cell_mask"]),
            batch["loss_mask"],
        ],
        dim=2,
    ).reshape(B, n_pairs * 2 * tk.CANVAS_SIZE).bool()
    loss_puzzle = torch.zeros(B, puzzle_len, dtype=torch.bool, device=device)
    loss = torch.cat([loss_puzzle, lm], dim=1)

    return token_ids, attention, loss