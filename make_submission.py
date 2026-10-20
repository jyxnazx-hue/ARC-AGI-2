import argparse
import json
from pathlib import Path
# pyrefly: ignore[missing-import]
import torch

from src.core import inference as inf
from train_trm import Config, SMOKE, REAL, build_model

def load_tasks(tasks_dir: Path, limit: int | None = None) -> dict:
    paths = sorted(tasks_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    return {p.stem: json.loads(p.read_text()) for p in paths}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["smoke", "real"], default="smoke")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--out", type=str, default="submission.json")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = SMOKE if args.config == "smoke" else REAL
    if args.device:
        cfg.device = args.device

    model = build_model(cfg).to(cfg.device)
    puzzle_idx_map = {}
    
    if args.ckpt:
        ckpt = torch.load(Path(args.ckpt), map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        puzzle_idx_map = ckpt.get("puzzle_idx_map", {})
        print(f"[load] {args.ckpt} step={ckpt.get('step', '?')}")

    tasks = load_tasks(Path(args.tasks))
    print(f"[tasks] {len(tasks)} loaded from {args.tasks}")

    inf.write_submission(
        model=model,
        tasks=tasks,
        puzzle_idx_map=puzzle_idx_map,
        n_train_pairs=cfg.n_train_pairs,
        puzzle_len=cfg.puzzle_len,
        out_path=args.out,
        device=cfg.device,
    )

if __name__ == "__main__":
    main()