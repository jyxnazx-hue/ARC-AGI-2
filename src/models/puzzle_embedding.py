import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class PuzzleEmbeddingLayer(nn.Module):
    def __init__(self, num_tasks: int = 1000, num_tokens: int = 16, d_model: int = 512):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.embeddings = nn.Parameter(torch.randn(num_tasks, num_tokens, d_model) * 0.02)
        self.unseen_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.task_to_idx = {}

    def register_tasks(self, task_ids: List[str]):
        for tid in task_ids:
            if tid not in self.task_to_idx:
                self.task_to_idx[tid] = len(self.task_to_idx)

    def forward(self, task_ids: List[str]) -> torch.Tensor:
        indices = [self.task_to_idx.get(tid, -1) for tid in task_ids]
        batch_embeds = [self.embeddings[idx] if idx >= 0 else self.unseen_embedding.squeeze(0) for idx in indices]
        return torch.stack(batch_embeds, dim=0)

    def init_unseen_puzzle_embedding(self, device: str = "cpu") -> torch.Tensor:
        return self.unseen_embedding.to(device)

    def prepend_embeddings(
        self,
        token_embeddings: torch.Tensor,
        task_ids: List[str],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_sz = token_embeddings.shape[0]
        device = token_embeddings.device
        prefix = self.forward(task_ids).to(device)
        full_embeddings = torch.cat([prefix, token_embeddings], dim=1)
        if attention_mask is not None:
            prefix_mask = torch.zeros((b_sz, self.num_tokens), dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            full_mask = torch.zeros((b_sz, full_embeddings.shape[1]), dtype=torch.bool, device=device)
        return full_embeddings, full_mask
