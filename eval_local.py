import os
import json
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import numpy as np
from pathlib import Path
from src.core import inference as inf
from train_trm import Config, build_model

DATA_ROOT = Path("data")
HOLDOUT_DIR = DATA_ROOT / "holdout"
CKPT_PATH = Path("checkpoints/trm_step1000.pt") 

def main():
    cfg = Config()
    cfg.device = "cpu"
    cfg.n_train_pairs = 2
    cfg.inner_steps_k = 4
    cfg.outer_steps_t = 3
    
    print(f"Loading {CKPT_PATH}...")
    model = build_model(cfg).to(cfg.device)
    ckpt = torch.load(CKPT_PATH, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded DRM model from step {ckpt.get('step', '?')}")

    paths = sorted(HOLDOUT_DIR.glob("*.json"))[:1] 
    task = json.loads(paths[0].read_text())
    print(f"Evaluating 1 holdout task: {paths[0].stem}...")

    a1, _ = inf.predict_two_attempts(
        model=model, 
        task=task, 
        puzzle_idx=model.n_puzzles, 
        n_train_pairs=cfg.n_train_pairs, 
        puzzle_len=cfg.puzzle_len,
        device=cfg.device,
        tau_schedule=[1.0, 0.5, 0.0] 
    )
    
    truth = np.array(task["test"][0]["output"])
    pred = np.array(a1)
    
    print("\n--- PREDICTION ---")
    print(pred)
    print("\n--- GROUND TRUTH ---")
    print(truth)
    
    if pred.shape == truth.shape:
        acc = np.sum(pred == truth) / truth.size
        print(f"\n Pixel Accuracy: {acc:.1%}")
    else:
        print("\n Shape mismatch.")

if __name__ == "__main__":
    main()
