from typing import Any, Optional, Dict
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import torch.nn as nn
# pyrefly: ignore[missing-import]
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, ffn_dim: int = 1024):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        kv = norm_x if context is None else self.norm1(context)
        attn_out, _ = self.attn(
            query=norm_x,
            key=kv,
            value=kv,
            key_padding_mask=attn_mask,
            need_weights=False,   
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SharedRecursiveNet(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = state
        for layer in self.layers:
            out = layer(out, context=context, attn_mask=key_padding_mask)
        return out


class TRMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 12,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_dim: int = 1024,
        inner_steps_k: int = 4,
        outer_steps_t: int = 3,
        n_puzzles: int = 800,
        puzzle_len: int = 8,
        n_pairs: int = 5,
        canvas_h: int = 30,
        canvas_w: int = 30,
    ):
        super().__init__()
        canvas_size = canvas_h * canvas_w

        self.d_model = d_model
        self.inner_steps_k = inner_steps_k
        self.outer_steps_t = outer_steps_t
        self.vocab_size = vocab_size
        self.n_puzzles = n_puzzles
        self.puzzle_len = puzzle_len
        self.n_pairs = n_pairs
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.canvas_size = canvas_size

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # Puzzle embeddings: n_puzzles real + 1 null
        self.puzzle_embeddings = nn.Embedding(
            n_puzzles + 1, puzzle_len * d_model
        )

        # 2D positional encoding
        self.row_embeddings = nn.Embedding(canvas_h, d_model)
        self.col_embeddings = nn.Embedding(canvas_w, d_model)

        # Init
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        nn.init.normal_(self.puzzle_embeddings.weight, std=0.02)
        nn.init.normal_(self.row_embeddings.weight, std=0.02)
        nn.init.normal_(self.col_embeddings.weight, std=0.02)
        with torch.no_grad():
            self.puzzle_embeddings.weight[n_puzzles].zero_()

        # Recursive network + decoder head
        self.shared_net = SharedRecursiveNet(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
        )
        self.decoder_head = nn.Linear(d_model, vocab_size, bias=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def expected_seq_len(self) -> int:
        return self.puzzle_len + self.n_pairs * 2 * self.canvas_size

    def _embed_inputs(
        self,
        token_ids: torch.Tensor,
        puzzle_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, T = token_ids.shape
        expected_T = self.expected_seq_len
        if T != expected_T:
            raise ValueError(f"expected T={expected_T}, got {T}")

        x = self.token_embeddings(token_ids)  # [B, T, d]

        p_idx = puzzle_idx.clamp(0, self.n_puzzles)
        p_emb = self.puzzle_embeddings(p_idx).view(
            B, self.puzzle_len, self.d_model
        )

        pos = torch.arange(self.canvas_size, device=token_ids.device)
        row = pos // self.canvas_w
        col = pos % self.canvas_w
        pos_emb = self.row_embeddings(row) + self.col_embeddings(col)

        n_canvases = self.n_pairs * 2
        pos_emb = pos_emb.repeat(n_canvases, 1)

        x_canvas = x[:, self.puzzle_len:] + pos_emb.unsqueeze(0)

        return torch.cat([p_emb, x_canvas], dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        puzzle_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tbptt_window: int = 4,
    ) -> Dict[str, Any]:
        x = self._embed_inputs(token_ids, puzzle_idx)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        if labels is None and loss_mask is not None:
            labels = torch.where(loss_mask.bool(), token_ids, -100)

        return self.forward_with_supervision(
            input_embeddings=x,
            labels=labels,
            key_padding_mask=key_padding_mask,
            tbptt_window=tbptt_window,
        )

    def forward_with_supervision(
        self,
        input_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        tbptt_window: int = 4,
    ) -> Dict[str, Any]:
        batch_size, seq_len, _ = input_embeddings.shape
        device = input_embeddings.device

        z_H = torch.randn(batch_size, seq_len, self.d_model, device=device) * 0.02
        z_L = torch.randn(batch_size, seq_len, self.d_model, device=device) * 0.02

        step_losses = []
        step_logits = []

        for step in range(self.outer_steps_t):
            if step > 0 and step % tbptt_window == 0:
                z_H = z_H.detach()
                z_L = z_L.detach()

            for _ in range(self.inner_steps_k):
                z_L = self.shared_net(
                    state=z_L,
                    context=z_H + input_embeddings,
                    key_padding_mask=key_padding_mask,
                )
            z_H = self.shared_net(
                state=z_H,
                context=z_L,
                key_padding_mask=key_padding_mask,
            )

            logits = self.decoder_head(z_H)
            step_logits.append(logits)

            if labels is not None:
                if (labels != -100).any():
                    loss_step = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
                else:
                    loss_step = logits.sum() * 0.0
                step_losses.append(loss_step)

        total_loss = torch.stack(step_losses).mean() if step_losses else None

        return {
            "total_loss": total_loss,
            "step_losses": step_losses,
            "final_logits": step_logits[-1],
            "all_logits": step_logits,
            "z_H": z_H,
            "z_L": z_L,
        }