import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.tokenizer import ARCTokenizer
from src.models.trm_core import TRMModel
from src.models.puzzle_embedding import PuzzleEmbeddingLayer
from src.models.train_trm import (
    ARCCanvasDataset,
    canvas_collate_fn,
    compute_canvas_loss,
    evaluate_pass_at_2_canvas,
    EMA,
    save_checkpoint
)

def run():
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    print(f"[Device] Accelerated training on: {torch.cuda.get_device_name(0)}")

    tokenizer = ARCTokenizer()
    build_dir = Path("data/build")
    holdout_dir = Path("data/holdout")

    # Ingest fixed-canvas dataset
    dataset = ARCCanvasDataset(build_dir, tokenizer, n_permutations=1)
    
    # Subsample 1/4th of the data to keep ~6,800 diverse canvas pairs for fast baseline verification
    indices = torch.arange(0, len(dataset), step=4)
    fast_dataset = torch.utils.data.Subset(dataset, indices)
    print(f"[Data] Fast-track dataset: {len(fast_dataset)} samples (subsampled from {len(dataset)})")

    physical_batch_size = 16
    grad_accum_steps = 2  # Effective batch size = 32

    loader = DataLoader(
        fast_dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        collate_fn=canvas_collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    model = TRMModel(
        vocab_size=12,
        d_model=512,
        n_heads=8,
        n_layers=2,
        ffn_dim=1024,
        inner_steps_k=2,
        outer_steps_t=2
    ).to(device)

    all_task_ids = [Path(f).stem for f in build_dir.glob("*.json")]
    puzzle_layer = PuzzleEmbeddingLayer(
        num_tasks=max(800, len(all_task_ids)),
        num_tokens=16,
        d_model=512
    ).to(device)
    puzzle_layer.register_tasks(all_task_ids)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 2e-4, "weight_decay": 0.1},
        {"params": [puzzle_layer.embeddings], "lr": 2e-4, "weight_decay": 0.01}
    ])

    scaler = GradScaler("cuda")
    ema = EMA(model, decay=0.999)
    epochs = 5
    total_steps = len(loader)

    print(f"[Train] Launching: {epochs} epochs | {total_steps} steps/epoch (Batch: {physical_batch_size * grad_accum_steps})\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, 1):
            in_c = batch["input_canvas"].to(device, non_blocking=True)
            tgt_c = batch["target_canvas"].to(device, non_blocking=True)
            mask = batch["loss_mask"].to(device, non_blocking=True)
            tids = batch["task_ids"]

            with autocast(device_type="cuda", dtype=torch.float16):
                c_emb = model.get_canvas_embeddings(in_c)
                full_emb, _ = puzzle_layer.prepend_embeddings(c_emb, tids)
                logits = model.forward_recursive(full_emb)["final_logits"]
                loss = compute_canvas_loss(logits, tgt_c, mask) / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0 or step == total_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_([puzzle_layer.embeddings], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            total_loss += (loss.item() * grad_accum_steps)

            if step % 50 == 0 or step == total_steps:
                elapsed = time.time() - start
                ms = (elapsed / step) * 1000.0
                print(f"Epoch [{epoch:02d}/{epochs:02d}] | Step [{step:03d}/{total_steps:03d}] | Loss: {(loss.item() * grad_accum_steps):.4f} | Speed: {ms:.1f}ms/step")

        avg_loss = total_loss / total_steps
        print(f"--> Epoch {epoch:02d} Complete | Mean Loss: {avg_loss:.4f} | Elapsed: {time.time() - start:.1f}s\n")

    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints/trm_canvas_baseline.pt")
    save_checkpoint(ckpt_path, model, puzzle_layer, optimizer, ema, step=epochs)
    print(f"[Done] Checkpoint saved to {ckpt_path.resolve()}")

    print("\n--- Running Final Step G Holdout Evaluation ---")
    score = evaluate_pass_at_2_canvas(model, puzzle_layer, tokenizer, holdout_dir, num_tasks=50, device="cuda")
    print(f">>> Holdout Pass@2: {score:.2f}% <<<")

if __name__ == "__main__":
    run()
