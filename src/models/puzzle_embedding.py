from typing import Dict, List, Optional
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn


class PuzzleEmbeddingLayer(nn.Module):
    def __init__(self, num_tasks: int = 800, num_tokens: int = 16, d_model: int = 512):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_tokens = num_tokens
        self.d_model = d_model

        self.task_to_idx: Dict[str, int] = {}
        self.embeddings = nn.Parameter(
            torch.randn(num_tasks, num_tokens, d_model) * 0.02
        )

    def register_tasks(self, task_ids: List[str]):
        self.task_to_idx = {t_id: idx for idx, t_id in enumerate(sorted(task_ids))}
        if len(self.task_to_idx) > self.num_tasks:
            raise ValueError(f"Task count {len(self.task_to_idx)} exceeds capacity {self.num_tasks}")

    def get_embedding(self, task_id: str) -> torch.Tensor:
        if task_id not in self.task_to_idx:
            raise KeyError(f"Task ID '{task_id}' not registered in puzzle embedding table.")
        idx = self.task_to_idx[task_id]
        return self.embeddings[idx]

    def forward(self, task_ids: List[str]) -> torch.Tensor:
        indices = [self.task_to_idx[tid] for tid in task_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.embeddings.device)
        return self.embeddings[idx_tensor]

    def prepend_embeddings(
        self,
        token_embeddings: torch.Tensor,
        task_ids: List[str],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        p_embeds = self.forward(task_ids)
        combined_embeddings = torch.cat([p_embeds, token_embeddings], dim=1)

        if attention_mask is not None:
            batch_size = token_embeddings.size(0)
            prefix_mask = torch.ones(
                (batch_size, self.num_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            return combined_embeddings, combined_mask

        return combined_embeddings

    def init_unseen_puzzle_embedding(self, device: Optional[torch.device] = None) -> nn.Parameter:
        target_device = device if device is not None else self.embeddings.device
        return nn.Parameter(
            torch.randn(1, self.num_tokens, self.d_model, device=target_device) * 0.02
        )
