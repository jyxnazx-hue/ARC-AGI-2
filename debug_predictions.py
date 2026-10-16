"""
debug_predictions.py
Inspects raw decoded tokens against ground-truth tokens directly.
"""

from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
from src.tokenizer import ARCTokenizer
from src.models.trm_core import TRMModel
from src.models.puzzle_embedding import PuzzleEmbeddingLayer
from src.models.train_trm import ARCTrainDataset, load_checkpoint

tokenizer = ARCTokenizer()
build_dir = Path("data/build")

dataset = ARCTrainDataset(
    tasks_dir=build_dir,
    tokenizer=tokenizer,
    n_permutations=1,
    max_seq_len=192,
    max_tasks=2
)

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
all_task_ids = [Path(f).stem for f in build_dir.glob("*.json")]
puzzle_layer.register_tasks(all_task_ids)

load_checkpoint(Path("checkpoints/trm_baseline.pt"), model, puzzle_layer)
model.eval()

# Check Sample 0
sample = dataset[0]
inp = torch.tensor([sample["token_ids"]], dtype=torch.long)
labels = torch.tensor(sample["labels"])
mask = (labels != -100)

with torch.no_grad():
    token_embeds = model.token_embeddings(inp)
    full_embeds, _ = puzzle_layer.prepend_embeddings(
        token_embeds, [sample["task_id"]], attention_mask=torch.zeros_like(inp, dtype=torch.bool)
    )
    outputs = model.forward_with_supervision(full_embeds)
    # Offset by 16 because of the prepended puzzle embedding tokens
    pred_logits = outputs["final_logits"][0, 16:]
    pred_tokens = pred_logits.argmax(dim=-1)

target_indices = torch.where(mask)[0]
print(f"Target Token Count: {len(target_indices)}")
correct = (pred_tokens[target_indices] == labels[target_indices]).sum().item()
print(f"Target Tokens Correct: {correct} / {len(target_indices)} ({correct / len(target_indices) * 100:.1f}%)")

print("\nSample Predicted vs Ground Truth:")
pred_decoded = tokenizer.decode(pred_tokens[target_indices][:15].tolist())
true_decoded = tokenizer.decode(labels[target_indices][:15].tolist())
print("Pred: ", pred_decoded)
print("True: ", true_decoded)