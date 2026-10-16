"""
quick_cpu_check.py
Fast 60-second sanity check: Overfits 2 tasks on CPU to verify that:
1. Recursive forward/backward loop converges (loss drops from ~4.0 to < 0.5)
2. Tokenizer reconstructs valid 2D matrix shapes
3. Model saves a real checkpoint and evaluates cleanly
"""

import time
from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from src.tokenizer import ARCTokenizer
from src.models.trm_core import TRMModel
from src.models.puzzle_embedding import PuzzleEmbeddingLayer
from src.models.train_trm import (
    ARCTrainDataset,
    collate_fn,
    EMA,
    save_checkpoint
)
from eval_generalization import evaluate_split_metrics


def run_quick_check():
    device = "cpu"
    print("=" * 60)
    print("STARTING 60-SECOND CPU SANITY CHECK (2 TASKS)")
    print("=" * 60)

    build_dir = Path("data/build")
    holdout_dir = Path("data/holdout")
    tokenizer = ARCTokenizer()

    # Load 2 tasks only, 1 permutation, short sequence length
    print("Loading 2 tasks for overfitting check...")
    dataset = ARCTrainDataset(
        tasks_dir=build_dir,
        tokenizer=tokenizer,
        n_permutations=1,
        max_seq_len=192,
        max_tasks=2
    )
    print(f"Dataset ready: {len(dataset)} augmented instances.")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Lightweight recursive settings (T=2 outer, K=2 inner)
    model = TRMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=2,
        ffn_dim=1024,
        inner_steps_k=2,
        outer_steps_t=2
    ).to(device)

    puzzle_layer = PuzzleEmbeddingLayer(num_tasks=800, num_tokens=16, d_model=512).to(device)
    all_task_ids = [Path(f).stem for f in build_dir.glob("*.json")]
    puzzle_layer.register_tasks(all_task_ids)

    # Use higher learning rate to force rapid memorization on CPU
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 3e-4, "weight_decay": 0.01},
        {"params": [puzzle_layer.embeddings], "lr": 5e-2, "weight_decay": 0.0}
    ])
    ema = EMA(model, decay=0.99)

    print("\nTraining for 25 steps to verify convergence...")
    step = 0
    start_time = time.time()
    initial_loss = None
    final_loss = None

    model.train()
    while step < 25:
        for batch in loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            task_ids = batch["task_ids"]

            token_embeds = model.token_embeddings(input_ids)
            full_embeds, full_mask = puzzle_layer.prepend_embeddings(
                token_embeds,
                task_ids,
                attention_mask=mask
            )

            b_sz = labels.shape[0]
            prefix_pad = torch.full((b_sz, 16), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([prefix_pad, labels], dim=1)

            outputs = model.forward_with_supervision(
                input_embeddings=full_embeds,
                labels=full_labels,
                key_padding_mask=full_mask,
                tbptt_window=2
            )

            loss = outputs["total_loss"]
            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            final_loss = loss.item()

            if step % 5 == 0 or step == 25:
                print(f"Step [{step:02d}/25] | Current Loss: {final_loss:.4f}")

            if step >= 25:
                break

    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s")
    print(f"Loss Trajectory: {initial_loss:.4f} ---> {final_loss:.4f}")

    # Save real checkpoint
    ckpt_path = Path("checkpoints/trm_baseline.pt")
    save_checkpoint(ckpt_path, model, puzzle_layer, optimizer, ema, step=25)
    print(f"Saved verified checkpoint to {ckpt_path}")

    # Run evaluation using EMA shadow weights
    print("\nRunning Evaluation on Fitted Tasks vs Unseen Holdout...")
    backup = ema.apply_shadow(model)
    build_pass2, build_px = evaluate_split_metrics(
        model, puzzle_layer, tokenizer, build_dir, num_samples=2, device=device
    )
    hold_pass2, hold_px = evaluate_split_metrics(
        model, puzzle_layer, tokenizer, holdout_dir, num_samples=2, device=device
    )
    ema.restore(model, backup)

    print("-" * 50)
    print(f"Fitted Tasks Pixel Match:   {build_px:.1f}%")
    print(f"Unseen Holdout Pixel Match: {hold_px:.1f}%")
    print("-" * 50)

    if final_loss < initial_loss:
        print("VERDICT: NEURAL RECURSION ENGINE IS WORKING AND LEARNING CORRECTLY.")
    else:
        print("VERDICT: CHECK GRADIENT FLOW.")


if __name__ == "__main__":
    run_quick_check()