# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import numpy as np
import math
import argparse
from pathlib import Path
import json

from src.tokenizer import ARCTokenizer
from src.models.trm_core import TRMModel
from src.models.puzzle_embedding import PuzzleEmbeddingLayer
from src.core import dataset as ds

# Configuration
VOCAB_SIZE = 12
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 2
FFN_DIM = 1024
INNER_STEPS_K = 2
OUTER_STEPS_T = 2
PUZZLE_LEN = 16
CANVAS_SIZE = 900
PAD_TOKEN = 0
MASK_TOKEN = 1
COLOR_OFFSET = 2

def load_model(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path}...")
    
    model = TRMModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_dim=FFN_DIM,
        inner_steps_k=INNER_STEPS_K,
        outer_steps_t=OUTER_STEPS_T
    ).to(device)

    # Initialize Puzzle Embeddings
    puzzle_layer = PuzzleEmbeddingLayer(
        num_tasks=800, 
        num_tokens=PUZZLE_LEN, 
        d_model=D_MODEL
    ).to(device)

    # Load Weights
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        puzzle_layer.load_state_dict(ckpt["puzzle_state"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    return model, puzzle_layer


def decode_canvas(canvas_tensor):
    """Converts a flat 900-token tensor to a 2D numpy grid, cropping out the PADs."""
    grid_1d = canvas_tensor.cpu().numpy()
    # Reshape to 30x30
    grid = grid_1d.reshape(30, 30)
    
    non_pad_mask = (grid != PAD_TOKEN)
    if not non_pad_mask.any():
        return np.zeros((20, 20), dtype=int)
        
    rows = np.any(non_pad_mask, axis=1)
    cols = np.any(non_pad_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped = grid[rmin:rmax+1, cmin:cmax+1]
    colors = np.where(cropped >= COLOR_OFFSET, cropped - COLOR_OFFSET, 0)
    return colors


def drm_inference(model, puzzle_layer, tokenizer, task_data, device, num_steps=16):
    train_inputs = []
    train_outputs = []
    for pair in task_data["train"]:
        in_grid = np.array(pair["input"])
        out_grid = np.array(pair["output"])
        in_canvas = np.zeros(900, dtype=np.int64)
        in_canvas[:in_grid.size] = in_grid.flatten() + COLOR_OFFSET
        train_inputs.append(in_canvas)
        
        out_canvas = np.zeros(900, dtype=np.int64)
        out_canvas[:out_grid.size] = out_grid.flatten() + COLOR_OFFSET
        train_outputs.append(out_canvas)

    # Test input
    test_in_grid = np.array(task_data["test"][0]["input"])
    test_in_canvas = np.zeros(900, dtype=np.int64)
    test_in_canvas[:test_in_grid.size] = test_in_grid.flatten() + COLOR_OFFSET
    
    # Initialize Test Output Canvas (All MASK)
    test_out_canvas = np.full(900, MASK_TOKEN, dtype=np.int64)
    
    # Create Tau Schedule (1.0 -> 0.0)
    taus = np.linspace(1.0, 0.0, num_steps)
    
    print(f"Starting DRM Remasking Loop ({num_steps} steps)...")
    
    for step, tau in enumerate(taus):
        seq = []
        seq.extend([PAD_TOKEN] * PUZZLE_LEN) 
        
        for i in range(len(train_inputs)):
            seq.extend(train_inputs[i])
            seq.extend(train_outputs[i])
            
        seq.extend(test_in_canvas)
        seq.extend(test_out_canvas)
        
        # Convert to tensor
        token_ids = torch.tensor([seq], dtype=torch.long, device=device)
        puzzle_idx = torch.tensor([0], dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model(token_ids, puzzle_idx, tau=torch.tensor([tau], device=device))
            logits = out["final_logits"][0, -CANVAS_SIZE:] 
            
        # Get predictions and confidence
        probs = torch.softmax(logits, dim=-1)
        
        probs[:, 0] = 0.0 # PAD
        probs[:, 1] = 0.0 # MASK
        
        pred = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Update canvas
        test_out_canvas = pred.cpu().numpy()
        
        # Remask lowest confidence tokens based on next tau
        if step < num_steps - 1:
            next_tau = taus[step + 1]
            # Cosine mask ratio
            mask_ratio = 1.0 - math.cos(next_tau * math.pi / 2)
            num_masked = int(CANVAS_SIZE * mask_ratio)
            num_keep = CANVAS_SIZE - num_masked
            
            if num_keep > 0:
                topk_indices = torch.topk(confidence, num_keep).indices
                new_canvas = np.full(900, MASK_TOKEN, dtype=np.int64)
                new_canvas[topk_indices.cpu().numpy()] = pred.cpu().numpy()[topk_indices.cpu().numpy()]
                test_out_canvas = new_canvas
                
        if step % 4 == 0:
            non_mask_count = (test_out_canvas != MASK_TOKEN).sum()
            print(f"  Step {step:02d} | Tau: {tau:.2f} | Masked: {900 - non_mask_count}/900")

    return torch.tensor(test_out_canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--task", type=str, required=True, help="Path to task .json file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    with open(args.task, "r") as f:
        task_data = json.load(f)
        
    # Load Model
    model, puzzle_layer = load_model(args.ckpt, device)
    
    # Run Inference
    final_tensor = drm_inference(model, puzzle_layer, None, task_data, device)
    
    # Decode
    pred_grid = decode_canvas(final_tensor)
    gt_grid = np.array(task_data["test"][0]["output"])
    
    print("\n--- PREDICTION ---")
    print(pred_grid)
    print("\n--- GROUND TRUTH ---")
    print(gt_grid)
    
    if pred_grid.shape == gt_grid.shape:
        accuracy = (pred_grid == gt_grid).mean() * 100
        print(f"\nExact Match: {np.array_equal(pred_grid, gt_grid)}")
        print(f"Pixel Accuracy: {accuracy:.1f}%")
    else:
        print(f"\nShape Mismatch! Pred: {pred_grid.shape}, GT: {gt_grid.shape}")

if __name__ == "__main__":
    main()
