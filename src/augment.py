import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

class D4SymmetryGroup:
    @staticmethod
    def apply_transform(grid: np.ndarray, transform_idx: int) -> np.ndarray:
        t = transform_idx % 8
        if t == 0: return grid.copy()
        elif t == 1: return np.rot90(grid, k=1).copy()
        elif t == 2: return np.rot90(grid, k=2).copy()
        elif t == 3: return np.rot90(grid, k=3).copy()
        elif t == 4: return np.fliplr(grid).copy()
        elif t == 5: return np.flipud(grid).copy()
        elif t == 6: return np.rot90(np.fliplr(grid), k=1).copy()
        elif t == 7: return np.rot90(np.flipud(grid), k=1).copy()
        return grid.copy()

class ColorPermuter:
    @staticmethod
    def generate_permutation() -> Dict[int, int]:
        perm = np.random.permutation(10)
        return {int(i): int(perm[i]) for i in range(10)}

    @staticmethod
    def apply_permutation(grid: np.ndarray, perm: Dict[int, int]) -> np.ndarray:
        res = grid.copy()
        for src, dst in perm.items():
            res[grid == src] = dst
        return res

def load_dataset_split(tasks_dir: Path) -> Dict[str, Dict[str, Any]]:
    tasks = {}
    for f in sorted(Path(tasks_dir).glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        train_pairs = [(np.array(p["input"], dtype=np.int64), np.array(p["output"], dtype=np.int64)) for p in data.get("train", [])]
        test_pairs = []
        for p in data.get("test", []):
            inp = np.array(p["input"], dtype=np.int64)
            out = np.array(p["output"], dtype=np.int64) if "output" in p else None
            test_pairs.append((inp, out))
        tasks[f.stem] = {"train": train_pairs, "test": test_pairs}
    return tasks

def augment_task_canvases(task: dict, tokenizer, n_permutations: int = 1) -> List[Dict[str, Any]]:
    train_pairs = task["train"]
    test_in = task["test"][0][0]
    test_out = task["test"][0][1] if task["test"][0][1] is not None else np.zeros((1, 1), dtype=np.int64)
    augmented_samples = []

    for p_in, p_out in train_pairs:
        for sym_idx in range(8):
            t_in = D4SymmetryGroup.apply_transform(p_in, sym_idx)
            t_out = D4SymmetryGroup.apply_transform(p_out, sym_idx)
            for _ in range(n_permutations):
                perm = ColorPermuter.generate_permutation()
                pt_in = ColorPermuter.apply_permutation(t_in, perm)
                pt_out = ColorPermuter.apply_permutation(t_out, perm)
                in_canvas, _ = tokenizer.grid_to_canvas(pt_in)
                out_canvas, out_mask = tokenizer.grid_to_canvas(pt_out)
                augmented_samples.append({
                    "input_canvas": in_canvas,
                    "target_canvas": out_canvas,
                    "loss_mask": out_mask,
                    "orig_shape": pt_out.shape
                })

    for sym_idx in range(8):
        t_in = D4SymmetryGroup.apply_transform(test_in, sym_idx)
        t_out = D4SymmetryGroup.apply_transform(test_out, sym_idx)
        for _ in range(n_permutations):
            perm = ColorPermuter.generate_permutation()
            pt_in = ColorPermuter.apply_permutation(t_in, perm)
            pt_out = ColorPermuter.apply_permutation(t_out, perm)
            in_canvas, _ = tokenizer.grid_to_canvas(pt_in)
            out_canvas, out_mask = tokenizer.grid_to_canvas(pt_out)
            augmented_samples.append({
                "input_canvas": in_canvas,
                "target_canvas": out_canvas,
                "loss_mask": out_mask,
                "orig_shape": pt_out.shape
            })

    return augmented_samples
