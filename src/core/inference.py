"""
inference loop + submission writer

Key design decisions:
  * Answer region shape: inferred from training pairs (all-same-shape ->
    same as input; else most common output shape; else fallback to input).
  * Test output canvas starts all-MASK inside the answer region.
  * Remasking loop with a descending tau schedule: at each step, forward,
    predict colors, then re-mask the lowest-confidence cells.
  * Two attempts per test input: identity + 180-degree rotation, so a
    flipped prediction can correct a broken upright one.
  * Unseen tasks use the null puzzle embedding (index = n_puzzles).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import torch.nn.functional as F

from src.core import tokenizer as tk
from src.core import augment as ag
from src.core import dataset as ds
from src.models.trm_core import TRMModel


# Descending: 1.0 (full mask) -> 0.0 (no mask)
TAU_SCHEDULE: List[float] = [
    1.0, 0.94, 0.87, 0.79, 0.71, 0.62, 0.53, 0.44,
    0.35, 0.27, 0.19, 0.12, 0.06, 0.02, 0.0,
]

# Answer shape heuristic

def infer_answer_shape(task: Dict[str, Any]) -> Tuple[int, int]:
    """
    Guess the (H, W) of the test output.
    Order of preference:
      1. If every train pair has in.shape == out.shape, use test input shape.
      2. If every train output has the same shape, use that.
      3. Most common train output shape.
    Clamped to [1, 30].
    """
    train_ins = [np.asarray(p["input"]).shape for p in task["train"]]
    train_outs = [np.asarray(p["output"]).shape for p in task["train"]]
    test_in = np.asarray(task["test"][0]["input"]).shape

    if all(i == o for i, o in zip(train_ins, train_outs)):
        h, w = test_in
    elif len(set(train_outs)) == 1:
        h, w = train_outs[0]
    else:
        h, w = Counter(train_outs).most_common(1)[0][0]

    return max(1, min(30, h)), max(1, min(30, w))

# Rotation helper (for attempt 2)

def _rot180_grid(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return np.rot90(np.asarray(grid, dtype=np.int64), k=2).tolist()


def _rot180_task(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "train": [
            {"input": _rot180_grid(p["input"]), "output": _rot180_grid(p["output"])}
            for p in task["train"]
        ],
        "test": [{"input": _rot180_grid(task["test"][0]["input"])}],
    }

# Single attempt

@torch.no_grad()
def predict_one_attempt(
    model: TRMModel,
    task: Dict[str, Any],
    puzzle_idx: int,
    n_train_pairs: int,
    puzzle_len: int,
    device: str = "cpu",
    tau_schedule: Sequence[float] = TAU_SCHEDULE,
) -> List[List[int]]:
    """
    Run one inference pass on `task`. Returns a list-of-lists grid.
    """
    model.eval()
    H, W = infer_answer_shape(task)

    # Build context canvases 
    n_total_pairs = n_train_pairs + 1
    input_canvases = np.zeros((n_total_pairs, tk.CANVAS_SIZE), dtype=np.int64)
    output_canvases = np.zeros((n_total_pairs, tk.CANVAS_SIZE), dtype=np.int64)
    input_cell_mask = np.zeros((n_total_pairs, tk.CANVAS_SIZE), dtype=np.int64)
    output_cell_mask = np.zeros((n_total_pairs, tk.CANVAS_SIZE), dtype=np.int64)
    pair_mask = np.zeros(n_total_pairs, dtype=np.int64)

    n_use = min(len(task["train"]), n_train_pairs)
    for slot in range(n_use):
        pair = task["train"][slot]
        in_c = ag.grid_to_canvas(pair["input"])
        out_c = ag.grid_to_canvas(pair["output"])
        input_canvases[slot] = tk.flatten(in_c)
        output_canvases[slot] = tk.flatten(out_c)
        input_cell_mask[slot] = tk.flatten(tk.pad_mask(in_c))
        output_cell_mask[slot] = tk.flatten(tk.pad_mask(out_c))
        pair_mask[slot] = 1

    test_in_c = ag.grid_to_canvas(task["test"][0]["input"])
    input_canvases[-1] = tk.flatten(test_in_c)
    input_cell_mask[-1] = tk.flatten(tk.pad_mask(test_in_c))
    pair_mask[-1] = 1

    # Answer region + initial all-MASK canvas 
    answer_region = np.zeros((tk.CANVAS_H, tk.CANVAS_W), dtype=bool)
    answer_region[:H, :W] = True
    answer_flat = answer_region.reshape(-1)
    answer_positions = np.where(answer_flat)[0]
    n_answer = int(answer_flat.sum())

    current_output = np.zeros(tk.CANVAS_SIZE, dtype=np.int64)
    current_output[answer_flat] = tk.MASK

    # Remasking loop 
    for i, tau in enumerate(tau_schedule):
        output_canvases[-1] = current_output
        output_cell_mask[-1] = answer_flat.astype(np.int64)

        batch = {
            "puzzle_idx": torch.tensor([puzzle_idx], dtype=torch.long),
            "input_canvases": torch.from_numpy(input_canvases).unsqueeze(0),
            "output_canvases": torch.from_numpy(output_canvases).unsqueeze(0),
            "pair_mask": torch.from_numpy(pair_mask).unsqueeze(0),
            "input_cell_mask": torch.from_numpy(input_cell_mask).unsqueeze(0),
            "output_cell_mask": torch.from_numpy(output_cell_mask).unsqueeze(0),
            "loss_mask": torch.zeros(
                1, n_total_pairs, tk.CANVAS_SIZE, dtype=torch.long
            ),
        }
        token_ids, attention, _ = ds.build_sequence(batch, puzzle_len=puzzle_len)
        token_ids = token_ids.to(device)
        attention = attention.to(device)
        pidx = batch["puzzle_idx"].to(device)

        out = model(token_ids, pidx, attention_mask=attention, loss_mask=None)
        test_logits = out["final_logits"][0, -tk.CANVAS_SIZE:]     # [900, 12]
        color_logits = test_logits[..., 2:12]                      # [900, 10]
        probs = F.softmax(color_logits, dim=-1)                    # [900, 10]
        pred = probs.argmax(dim=-1) + 2                            # [900]
        conf = probs.max(dim=-1).values                            # [900]

        pred_np = pred.cpu().numpy()
        current_output[answer_flat] = pred_np[answer_flat]

        # Remask based on next tau (keep r(τ) = 1 - τ)
        if i < len(tau_schedule) - 1:
            next_tau = tau_schedule[i + 1]
            keep_frac = 1.0 - next_tau
            n_keep = max(1, int(round(keep_frac * n_answer)))

            conf_in_answer = conf[torch.from_numpy(answer_positions)]
            topk_idx = conf_in_answer.topk(n_keep).indices.cpu().numpy()
            keep_mask = np.zeros(n_answer, dtype=bool)
            keep_mask[topk_idx] = True

            # Mask the non-kept positions
            mask_positions = answer_positions[~keep_mask]
            current_output[mask_positions] = tk.MASK

    # --- Crop to (H, W) and undo color offset ---
    final_canvas = tk.unflatten(current_output)
    final = final_canvas[:H, :W] - tk.COLOR_OFFSET
    final = np.where(final < 0, 0, final)
    return final.tolist()

# Two attempts

def predict_two_attempts(
    model: TRMModel,
    task: Dict[str, Any],
    puzzle_idx: int,
    n_train_pairs: int,
    puzzle_len: int,
    device: str = "cpu",
    tau_schedule: Sequence[float] = TAU_SCHEDULE,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Attempt 1: identity. Attempt 2: 180-degree rotation, then un-rotate."""
    a1 = predict_one_attempt(
        model, task, puzzle_idx, n_train_pairs, puzzle_len, device, tau_schedule
    )
    task_180 = _rot180_task(task)
    a2_rot = predict_one_attempt(
        model, task_180, puzzle_idx, n_train_pairs, puzzle_len, device, tau_schedule
    )
    a2 = _rot180_grid(a2_rot)
    return a1, a2

# Submission writer

def write_submission(
    model: TRMModel,
    tasks: Dict[str, Dict[str, Any]],
    puzzle_idx_map: Optional[Dict[str, int]],
    n_train_pairs: int,
    puzzle_len: int,
    out_path: str | Path,
    device: str = "cpu",
    tau_schedule: Sequence[float] = TAU_SCHEDULE,
    verbose: bool = True,
) -> Path:
    submission: Dict[str, Dict[str, List[List[int]]]] = {}
    null_idx = model.n_puzzles

    for i, (task_id, task) in enumerate(tasks.items(), 1):
        pidx = (puzzle_idx_map or {}).get(task_id, null_idx)
        a1, a2 = predict_two_attempts(
            model, task, pidx, n_train_pairs, puzzle_len, device, tau_schedule
        )
        submission[task_id] = {"attempt_1": a1, "attempt_2": a2}
        if verbose and i % 10 == 0:
            print(f"[infer] {i}/{len(tasks)} tasks done")

    out_path = Path(out_path)
    out_path.write_text(json.dumps(submission))
    if verbose:
        print(f"[infer] wrote {out_path}  ({len(submission)} tasks)")
    return out_path

# Checkpoint helper

def load_checkpoint_for_inference(
    ckpt_path: str | Path,
    device: str = "cpu",
) -> Tuple[TRMModel, Dict[str, int], Dict[str, Any]]:
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)

    # Infer model config from the state dict
    sd = ckpt["model"]
    token_emb_w = sd["token_embeddings.weight"]
    vocab_size, d_model = token_emb_w.shape

    puzzle_w = sd["puzzle_embeddings.weight"]
    n_puzzles_plus_1, puzzle_len_x_d = puzzle_w.shape
    n_puzzles = n_puzzles_plus_1 - 1
    puzzle_len = puzzle_len_x_d // d_model

    cfg_name = ckpt.get("config_name", "unknown")
    meta = {"config_name": cfg_name, "step": ckpt["step"]}
    return (
        (vocab_size, d_model, n_puzzles, puzzle_len),
        ckpt.get("puzzle_idx_map", {}),
        meta,
    )