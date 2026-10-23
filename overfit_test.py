import os
import sys
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import torch.nn.functional as F

# Fix path for local execution
sys.path.insert(0, os.path.abspath("."))

from src.core import dataset as ds
from train_trm import Config, build_model
from src.corrupt import corrupt_canvas

# 1. Setup
cfg = Config()
cfg.device = "cpu"
cfg.n_train_pairs = 2
cfg.inner_steps_k = 4
cfg.outer_steps_t = 3

# 2. Load Train Data
train_ds = ds.ARCTrainDataset(
    ["data/build/00576224.json"], 
    puzzle_idx_map={"00576224": 0}, 
    n_train_pairs=cfg.n_train_pairs, 
    n_color_perms=0, 
    augment=False
)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=1, collate_fn=ds.collate_fn)
batch = next(iter(dataloader))

# 3. Create Model & Optimizer (Lower learning rate)
model = build_model(cfg).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

clean_token_ids, _, _ = ds.build_sequence(batch, puzzle_len=cfg.puzzle_len)
clean_token_ids = clean_token_ids.to(cfg.device)
puzzle_idx = torch.tensor([0]).to(cfg.device)

print("Starting Overfit Test (500 steps) on CPU...")
for step in range(501):
    model.train()
    optimizer.zero_grad()
    
    tau = torch.rand(1, device=cfg.device)
    corrupted_outputs = batch["output_canvases"].clone()
    corrupted_outputs[0, -1] = corrupt_canvas(
        canvas=batch["output_canvases"][0, -1],
        cell_mask=batch["output_cell_mask"][0, -1],
        tau=tau[0].item()
    )
    
    corrupted_batch = {**batch, "output_canvases": corrupted_outputs}
    token_ids, attention, loss_mask = ds.build_sequence(corrupted_batch, puzzle_len=cfg.puzzle_len)
    
    token_ids = token_ids.to(cfg.device)
    attention = attention.to(cfg.device)
    loss_mask = loss_mask.to(cfg.device)
    
    out = model(token_ids, puzzle_idx, attention_mask=attention, loss_mask=loss_mask, tau=tau)
    logits = out["final_logits"]
    
    masked_targets = torch.where(loss_mask, clean_token_ids, -100)
    
    # NO CLASS WEIGHTS. Just standard Cross Entropy.
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.view(-1), ignore_index=-100)
    
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

print("\nTesting prediction...")
model.eval()
with torch.no_grad():
    test_out = torch.full_like(batch["output_canvases"][0, -1], 1)
    corrupted_batch = {**batch, "output_canvases": batch["output_canvases"].clone()}
    corrupted_batch["output_canvases"][0, -1] = test_out
    
    token_ids, attention, _ = ds.build_sequence(corrupted_batch, puzzle_len=cfg.puzzle_len)
    tau_tensor = torch.tensor([1.0]).to(cfg.device)
    
    out = model(token_ids.to(cfg.device), puzzle_idx, attention_mask=attention.to(cfg.device), loss_mask=None, tau=tau_tensor)
    
    # CORRECT DECODING LOGIC
    color_logits = out["final_logits"][0, -900:][..., 2:12]
    preds = color_logits.argmax(dim=-1) + 2
    preds = preds.cpu().numpy().reshape(30, 30)
    
    truth = batch["output_canvases"][0, -1].cpu().numpy().reshape(30, 30)
    
    print("\n--- Predicted Grid (Top-Left 5x5) ---")
    print(preds[:5, :5])
    print("\n--- True Grid (Top-Left 5x5) ---")
    print((truth - 2)[:5, :5])