from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch

from src.tokenizer import ARCTokenizer
from src.augment import load_dataset_split
from src.models.trm_core import TRMModel
from src.models.puzzle_embedding import PuzzleEmbeddingLayer
from src.models.train_trm import load_checkpoint


def evaluate_split_metrics(
    model: TRMModel,
    puzzle_layer: PuzzleEmbeddingLayer,
    tokenizer: ARCTokenizer,
    tasks_dir: Path,
    num_samples: int = 20,
    device: str = "cpu"
):
    model.eval()
    tasks = load_dataset_split(tasks_dir)
    selected_keys = list(tasks.keys())[:num_samples]

    exact_matches = 0
    pixel_accuracies = []

    with torch.no_grad():
        for tid in selected_keys:
            task = tasks[tid]
            prompt_toks, _ = tokenizer.serialize_task(task, test_idx=0, mode="row_by_row", include_target=False)
            prompt_ids = torch.tensor([tokenizer.encode(prompt_toks)], dtype=torch.long, device=device)
            embeds = model.token_embeddings(prompt_ids)

            if tid in puzzle_layer.task_to_idx:
                p_emb = puzzle_layer([tid]).to(device)
            else:
                p_emb = puzzle_layer.init_unseen_puzzle_embedding(device=device)

            combined_embeds = torch.cat([p_emb, embeds], dim=1)
            outputs = model.forward_with_supervision(combined_embeds, labels=None)
            logits = outputs["final_logits"]

            # Generate Top-1 and Top-2 candidate grids
            pred_id_top1 = logits.argmax(dim=-1)[0].tolist()
            pred_id_top2 = logits.topk(2, dim=-1).indices[0, :, 1].tolist()

            pred_grid1 = tokenizer.tokens_to_grid(tokenizer.decode(pred_id_top1))
            pred_grid2 = tokenizer.tokens_to_grid(tokenizer.decode(pred_id_top2))

            gt_grid = task["test"][0][1]
            if gt_grid is None:
                continue

            # Exact match check
            m1 = np.array_equal(pred_grid1, gt_grid)
            m2 = np.array_equal(pred_grid2, gt_grid)
            if m1 or m2:
                exact_matches += 1

            # Per-pixel accuracy on top-1 prediction
            if pred_grid1.shape == gt_grid.shape:
                acc = (pred_grid1 == gt_grid).mean()
            else:
                acc = 0.0
            pixel_accuracies.append(acc)

    pass_at_2 = (exact_matches / len(selected_keys)) * 100
    mean_pixel_acc = float(np.mean(pixel_accuracies)) * 100 if pixel_accuracies else 0.0

    return pass_at_2, mean_pixel_acc


def run_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = ARCTokenizer()

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
    build_dir = Path("data/build")
    all_task_ids = [Path(f).stem for f in build_dir.glob("*.json")]
    puzzle_layer.register_tasks(all_task_ids)

    ckpt_path = Path("checkpoints/trm_baseline.pt")
    if ckpt_path.exists():
        print(f"Loading trained weights from {ckpt_path}...")
        load_checkpoint(ckpt_path, model, puzzle_layer)
    else:
        print("No checkpoint found; running diagnostic on active model state.")

    print("\n--- Evaluating Build Split (Training Distribution) ---")
    build_pass2, build_px = evaluate_split_metrics(
        model, puzzle_layer, tokenizer, Path("data/build"), num_samples=20, device=device
    )
    print(f"Build Pass@2:       {build_pass2:.1f}%")
    print(f"Build Pixel Match:  {build_px:.1f}%")

    print("\n--- Evaluating Holdout Split (Unseen Validation) ---")
    hold_pass2, hold_px = evaluate_split_metrics(
        model, puzzle_layer, tokenizer, Path("data/holdout"), num_samples=20, device=device
    )
    print(f"Holdout Pass@2:      {hold_pass2:.1f}%")
    print(f"Holdout Pixel Match: {hold_px:.1f}%")

    print("\n" + "=" * 50)
    print("GENERALIZATION ANALYSIS")
    print("=" * 50)
    gap = build_px - hold_px
    print(f"Pixel Accuracy Gap: {gap:.1f}%")
    if gap > 40.0:
        print("Status: Overfitting detected. Train accuracy significantly outpaces holdout.")
    elif hold_pass2 >= 5.0 or hold_px >= 70.0:
        print("Status: Generalization verified. Model retains rule transfer on unseen holdout.")
    else:
        print("Status: Under-adapted baseline. Online Test-Time Training (TTT) is required for unseen tasks.")
    print("=" * 50)


if __name__ == "__main__":
    run_check()