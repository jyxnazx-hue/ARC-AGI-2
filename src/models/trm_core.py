import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RecurrentTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, ffn_dim: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        B, S, _ = x.shape
        normed_q = self.norm1(x)
        normed_kv = self.norm1(context) if context is not None else normed_q
        S_kv = normed_kv.shape[1]

        q = self.q_proj(normed_q).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_kv).view(B, S_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_kv).view(B, S_kv, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

class TRMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 12,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_dim: int = 1024,
        inner_steps_k: int = 2,
        outer_steps_t: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.inner_steps_k = inner_steps_k
        self.outer_steps_t = outer_steps_t
        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.row_embed = nn.Embedding(30, d_model)
        self.col_embed = nn.Embedding(30, d_model)

        rows = torch.arange(30).unsqueeze(1).expand(30, 30).flatten()
        cols = torch.arange(30).unsqueeze(0).expand(30, 30).flatten()
        self.register_buffer("grid_rows", rows)
        self.register_buffer("grid_cols", cols)

        self.layers = nn.ModuleList([
            RecurrentTransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def get_canvas_embeddings(self, canvas_tokens: torch.Tensor) -> torch.Tensor:
        tok_embeds = self.token_embeddings(canvas_tokens)
        pos_embeds = self.row_embed(self.grid_rows) + self.col_embed(self.grid_cols)
        return tok_embeds + pos_embeds.unsqueeze(0)

    def forward_recursive(self, input_embeddings: torch.Tensor, tbptt_window: int = 2) -> dict:
        z_H = torch.zeros_like(input_embeddings)
        z_L = torch.zeros_like(input_embeddings)

        for t in range(self.outer_steps_t):
            for k in range(self.inner_steps_k):
                ctx = z_H + input_embeddings
                for layer in self.layers:
                    z_L = layer(z_L, context=ctx)
            for layer in self.layers:
                z_H = layer(z_H, context=z_L)

            if (t + 1) % tbptt_window == 0 and (t + 1) < self.outer_steps_t:
                z_H = z_H.detach()
                z_L = z_L.detach()

        normed = self.final_norm(z_H)
        logits = self.head(normed)
        return {"final_logits": logits}
