from __future__ import annotations
import math
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import ARCTokenizer
from src.augment import load_dataset_split, augment_task
from src.models.trm_core import TRMModel
from src.models.puzzle_embedding import PuzzleEmbeddingLayer


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].data.mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.clone().detach()
                param.data.copy_(self.shadow[name].data)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name].data)


class ARCTrainDataset(Dataset):
    def __init__(
        self,
        tasks_dir: Path,
        tokenizer: ARCTokenizer,
        n_permutations: int = 1,
        max_seq_len: int = 512,
        max_tasks: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        all_tasks = load_dataset_split(tasks_dir)
        task_items = list(all_tasks.items())
        if max_tasks is not None:
            task_items = task_items[:max_tasks]

        for _, task in task_items:
            augmented = augment_task(task, n_permutations=n_permutations, seed=42)
            for ex in augmented:
                for test_idx in range(len(ex["test"])):
                    # Extract prompt and target token sequences
                    mode = ex["mode"]
                    prompt_toks, target_toks = self.tokenizer.serialize_task(
                        task=task,
                        test_idx=test_idx,
                        mode=mode,
                        include_target=True
                    )
                    if target_toks is None:
                        continue

                    # Concatenate prompt + target tokens
                    full_tokens = prompt_toks + target_toks
                    token_ids = self.tokenizer.encode(full_tokens)

                    # Prompt tokens receive label -100 (masked from cross entropy)
                    labels = [-100] * len(prompt_toks)
                    target_ids = self.tokenizer.encode(target_toks)
                    labels.extend(target_ids)

                    # Truncate to max_seq_len
                    token_ids = token_ids[:max_seq_len]
                    labels = labels[:max_seq_len]

                    self.samples.append({
                        "task_id": task["id"],
                        "token_ids": token_ids,
                        "labels": labels
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    task_ids = [b["task_id"] for b in batch]
    max_len = max(len(b["token_ids"]) for b in batch)

    batch_input_ids = []
    batch_labels = []
    batch_mask = []

    for b in batch:
        seq_len = len(b["token_ids"])
        pad_len = max_len - seq_len

        input_ids = b["token_ids"] + [pad_id] * pad_len
        labels = b["labels"] + [-100] * pad_len
        # Key padding mask: True indicates masked (padded) positions
        mask = [False] * seq_len + [True] * pad_len

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_mask.append(mask)

    return {
        "task_ids": task_ids,
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.bool)
    }


def save_checkpoint(
    path: Path,
    model: TRMModel,
    puzzle_layer: PuzzleEmbeddingLayer,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    step: int
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "puzzle_state": puzzle_layer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "ema_shadow": ema.shadow
    }, path)


def load_checkpoint(
    path: Path,
    model: TRMModel,
    puzzle_layer: PuzzleEmbeddingLayer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMA] = None
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    puzzle_layer.load_state_dict(checkpoint["puzzle_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if ema and "ema_shadow" in checkpoint:
        ema.shadow = checkpoint["ema_shadow"]
    return checkpoint["step"]


def evaluate_pass_at_2(
    model: TRMModel,
    puzzle_layer: PuzzleEmbeddingLayer,
    tokenizer: ARCTokenizer,
    eval_tasks_dir: Path,
    max_eval_tasks: int = 10,
    device: str = "cpu"
) -> float:
    model.eval()
    tasks = load_dataset_split(eval_tasks_dir)
    task_keys = list(tasks.keys())[:max_eval_tasks]

    solved = 0
    total = len(task_keys)

    with torch.no_grad():
        for tid in task_keys:
            task = tasks[tid]
            prompt_toks, _ = tokenizer.serialize_task(task, test_idx=0, mode="row_by_row", include_target=False)
            prompt_ids = torch.tensor([tokenizer.encode(prompt_toks)], dtype=torch.long, device=device)
            embeds = model.token_embeddings(prompt_ids)

            # Get unseen or registered puzzle embedding
            if tid in puzzle_layer.task_to_idx:
                p_emb = puzzle_layer([tid]).to(device)
            else:
                p_emb = puzzle_layer.init_unseen_puzzle_embedding(device=device)

            combined_embeds = torch.cat([p_emb, embeds], dim=1)
            outputs = model.forward_with_supervision(combined_embeds, labels=None)
            logits = outputs["final_logits"]

            # Decode top-2 predictions from greedy & top-2 token paths
            pred_id_top1 = logits.argmax(dim=-1)[0].tolist()
            pred_id_top2 = logits.topk(2, dim=-1).indices[0, :, 1].tolist()

            pred_grid1 = tokenizer.tokens_to_grid(tokenizer.decode(pred_id_top1))
            pred_grid2 = tokenizer.tokens_to_grid(tokenizer.decode(pred_id_top2))

            gt_grid = task["test"][0][1]
            if gt_grid is not None:
                if np.array_equal(pred_grid1, gt_grid) or np.array_equal(pred_grid2, gt_grid):
                    solved += 1

    return (solved / total) if total > 0 else 0.0
