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
    save_checkpoint,
    evaluate_pass_at_2
)


def run_training(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    puzzle_lr: float = 1e-2,
    max_train_tasks: int = 20,   # Set to None for full 800-task run
    max_eval_tasks: int = 5      # Fast validation subset
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    build_dir = Path("data/build")
    holdout_dir = Path("data/holdout")

    tokenizer = ARCTokenizer()

    print(f"Building training dataset (max_tasks={max_train_tasks})...")
    dataset = ARCTrainDataset(
        tasks_dir=build_dir,
        tokenizer=tokenizer,
        n_permutations=1,
        max_seq_len=256,
        max_tasks=max_train_tasks
    )
    print(f"Total training samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = TRMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=2,
        ffn_dim=1024,
        inner_steps_k=2,
        outer_steps_t=2
    ).to(device)

    puzzle_layer = PuzzleEmbeddingLayer(
        num_tasks=800,
        num_tokens=16,
        d_model=512
    ).to(device)

    all_task_ids = [Path(f).stem for f in build_dir.glob("*.json")]
    puzzle_layer.register_tasks(all_task_ids)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr, "weight_decay": 0.1},
        {"params": [puzzle_layer.embeddings], "lr": puzzle_lr, "weight_decay": 1.0}
    ])

    ema = EMA(model, decay=0.999)

    print(f"Starting training: {epochs} epochs, {len(loader)} steps per epoch...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(loader, 1):
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()

            if step % 10 == 0 or step == len(loader):
                print(f"Epoch [{epoch}/{epochs}] | Step [{step}/{len(loader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # Evaluate Pass@2 on holdout
    print("\nRunning Holdout Evaluation...")
    backup = ema.apply_shadow(model)
    score = evaluate_pass_at_2(
        model=model,
        puzzle_layer=puzzle_layer,
        tokenizer=tokenizer,
        eval_tasks_dir=holdout_dir,
        max_eval_tasks=max_eval_tasks,
        device=device
    )
    ema.restore(model, backup)
    print(f">>> Final Pass@2 on Holdout Subset: {score * 100:.2f}% <<<")

    ckpt_path = Path("checkpoints/trm_baseline.pt")
    save_checkpoint(ckpt_path, model, puzzle_layer, optimizer, ema, step=epochs)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    run_training()
