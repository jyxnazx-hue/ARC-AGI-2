from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    load_checkpoint,
    evaluate_pass_at_2
)


def test_components_7_and_8():
    tokenizer = ARCTokenizer()
    build_dir = Path("data/build")
    holdout_dir = Path("data/holdout")

    # 1. Dataset & DataLoader integration test
    dataset = ARCTrainDataset(
        tasks_dir=build_dir,
        tokenizer=tokenizer,
        n_permutations=1,
        max_seq_len=256,
        max_tasks=5
    )
    assert len(dataset) > 0

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape

    # 2. Model & Optimizer with EMA test
    model = TRMModel(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=2,
        ffn_dim=1024,
        inner_steps_k=2,
        outer_steps_t=2
    )
    puzzle_layer = PuzzleEmbeddingLayer(num_tasks=800, num_tokens=16, d_model=512)
    puzzle_layer.register_tasks([b["task_id"] for b in dataset.samples])

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [puzzle_layer.embeddings],
        lr=1e-4,
        weight_decay=0.1
    )
    ema = EMA(model, decay=0.999)

    # 3. Single training forward-backward optimization step
    token_embeds = model.token_embeddings(batch["input_ids"])
    full_embeds, full_mask = puzzle_layer.prepend_embeddings(
        token_embeds,
        batch["task_ids"],
        attention_mask=batch["attention_mask"]
    )

    # Prepend dummy -100 labels for the 16 puzzle prefix positions
    batch_size = batch["labels"].shape[0]
    puzzle_pad_labels = torch.full((batch_size, 16), -100, dtype=torch.long)
    full_labels = torch.cat([puzzle_pad_labels, batch["labels"]], dim=1)

    outputs = model.forward_with_supervision(
        input_embeddings=full_embeds,
        labels=full_labels,
        key_padding_mask=full_mask
    )

    loss = outputs["total_loss"]
    assert loss is not None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ema.update(model)
    optimizer.zero_grad()

    # 4. Checkpoint saving and loading verification
    ckpt_path = Path("tests/temp_checkpoint.pt")
    save_checkpoint(ckpt_path, model, puzzle_layer, optimizer, ema, step=1)
    assert ckpt_path.exists()

    loaded_step = load_checkpoint(ckpt_path, model, puzzle_layer, optimizer, ema)
    assert loaded_step == 1
    if ckpt_path.exists():
        ckpt_path.unlink()

    # 5. Pass@2 Evaluator execution test
    pass_at_2 = evaluate_pass_at_2(
        model=model,
        puzzle_layer=puzzle_layer,
        tokenizer=tokenizer,
        eval_tasks_dir=holdout_dir,
        max_eval_tasks=2
    )
    assert 0.0 <= pass_at_2 <= 1.0

    print("COMPONENTS 7 & 8 (TRAINING LOOP, EMA, CHECKPOINTING, PASS@2 EVAL) VERIFIED")


if __name__ == "__main__":
    test_components_7_and_8()