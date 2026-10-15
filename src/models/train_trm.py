import copy
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.augment import load_dataset_split, augment_task_canvases

class ARCCanvasDataset(Dataset):
    def __init__(self, tasks_dir: Path, tokenizer, n_permutations: int = 1, max_tasks: Optional[int] = None):
        self.tokenizer = tokenizer
        tasks = load_dataset_split(tasks_dir)
        task_ids = sorted(list(tasks.keys()))
        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]

        self.samples = []
        for tid in task_ids:
            task_augs = augment_task_canvases(tasks[tid], tokenizer, n_permutations=n_permutations)
            for aug in task_augs:
                self.samples.append({
                    "task_id": tid,
                    "input_canvas": aug["input_canvas"],
                    "target_canvas": aug["target_canvas"],
                    "loss_mask": aug["loss_mask"],
                    "orig_shape": aug["orig_shape"]
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "task_id": s["task_id"],
            "input_canvas": torch.tensor(s["input_canvas"], dtype=torch.long),
            "target_canvas": torch.tensor(s["target_canvas"], dtype=torch.long),
            "loss_mask": torch.tensor(s["loss_mask"], dtype=torch.float32)
        }

def canvas_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "task_ids": [b["task_id"] for b in batch],
        "input_canvas": torch.stack([b["input_canvas"] for b in batch]),
        "target_canvas": torch.stack([b["target_canvas"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch])
    }

def compute_canvas_loss(logits: torch.Tensor, target_canvas: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    canvas_logits = logits[:, 16:, :]
    vocab_size = canvas_logits.shape[-1]
    flat_logits = canvas_logits.reshape(-1, vocab_size)
    flat_targets = target_canvas.reshape(-1)
    flat_mask = loss_mask.reshape(-1)

    ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    masked_loss = (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
    return masked_loss

def evaluate_pass_at_2_canvas(
    model,
    puzzle_layer,
    tokenizer,
    tasks_dir: Path,
    num_tasks: int = 10,
    device: str = "cuda"
) -> float:
    model.eval()
    tasks = load_dataset_split(tasks_dir)
    selected_keys = list(tasks.keys())[:num_tasks]

    solved = 0
    with torch.no_grad():
        for tid in selected_keys:
            task = tasks[tid]
            test_in = task["test"][0][0]
            gt_grid = task["test"][0][1]
            if gt_grid is None:
                continue

            in_canvas, _ = tokenizer.grid_to_canvas(test_in)
            in_tensor = torch.from_numpy(in_canvas).unsqueeze(0).to(device=device, dtype=torch.long)
            c_emb = model.get_canvas_embeddings(in_tensor)

            if tid in puzzle_layer.task_to_idx:
                p_emb = puzzle_layer([tid]).to(device)
            else:
                p_emb = puzzle_layer.init_unseen_puzzle_embedding(device=device)

            full_emb = torch.cat([p_emb, c_emb], dim=1)
            logits = model.forward_recursive(full_emb)["final_logits"][:, 16:, :]

            top1_tokens = logits.argmax(dim=-1)[0].cpu().numpy()
            top2_tokens = logits.topk(2, dim=-1).indices[0, :, 1].cpu().numpy()

            h, w = gt_grid.shape
            pred1 = tokenizer.canvas_to_grid(top1_tokens, h=h, w=w)
            pred2 = tokenizer.canvas_to_grid(top2_tokens, h=h, w=w)

            if np.array_equal(pred1, gt_grid) or np.array_equal(pred2, gt_grid):
                solved += 1

    return (solved / len(selected_keys)) * 100.0 if selected_keys else 0.0

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].copy_(
                        self.decay * self.shadow[name] + (1.0 - self.decay) * param
                    )

def save_checkpoint(path: Path, model, puzzle_layer, optimizer, ema, step: int = 0):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "ema_state": ema.shadow,
        "puzzle_state": puzzle_layer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
