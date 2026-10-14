import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
# pyrefly: ignore [missing-import]
import numpy as np


def load_task(task_path: Path) -> Dict[str, Any]:
    task_path = Path(task_path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    with open(task_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    task_data = {"id": task_path.stem, "train": [], "test": []}

    for pair in raw_data.get("train", []):
        inp = np.array(pair["input"], dtype=np.int8)
        out = np.array(pair["output"], dtype=np.int8)
        if inp.ndim != 2 or out.ndim != 2:
            raise ValueError(f"Grids must be 2D. Got inp: {inp.shape}, out: {out.shape}")
        task_data["train"].append((inp, out))

    for pair in raw_data.get("test", []):
        inp = np.array(pair["input"], dtype=np.int8)
        out = np.array(pair["output"], dtype=np.int8) if "output" in pair else None
        task_data["test"].append((inp, out))

    return task_data


def load_dataset_split(split_dir: Path) -> Dict[str, Dict[str, Any]]:
    split_dir = Path(split_dir)
    tasks = {}
    for f in sorted(split_dir.glob("*.json")):
        tasks[f.stem] = load_task(f)
    return tasks


D4_TRANSFORMS = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose"
]

D4_INVERSES = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose"
}


def apply_d4(grid: np.ndarray, transform_id: str) -> np.ndarray:
    if transform_id == "identity":
        return grid.copy()
    elif transform_id == "rot90":
        return np.rot90(grid, k=1)
    elif transform_id == "rot180":
        return np.rot90(grid, k=2)
    elif transform_id == "rot270":
        return np.rot90(grid, k=3)
    elif transform_id == "flip_h":
        return np.fliplr(grid)
    elif transform_id == "flip_v":
        return np.flipud(grid)
    elif transform_id == "transpose":
        return np.transpose(grid)
    elif transform_id == "anti_transpose":
        return np.flipud(np.rot90(grid, k=1))
    else:
        raise ValueError(f"Unknown transform: '{transform_id}'")


def invert_d4(grid: np.ndarray, transform_id: str) -> np.ndarray:
    if transform_id not in D4_INVERSES:
        raise ValueError(f"Unknown transform: '{transform_id}'")
    inverse_op = D4_INVERSES[transform_id]
    return apply_d4(grid, inverse_op)


def permute_colors(task: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    mapping = rng.permutation(10).astype(np.int8)

    permuted_task = {"id": task["id"], "train": [], "test": []}

    for inp, out in task["train"]:
        p_inp = mapping[inp]
        p_out = mapping[out]
        permuted_task["train"].append((p_inp, p_out))

    for inp, out in task["test"]:
        p_inp = mapping[inp]
        p_out = mapping[out] if out is not None else None
        permuted_task["test"].append((p_inp, p_out))

    return permuted_task


def serialize(grid: np.ndarray, mode: str = "row_by_row") -> List[str]:
    tokens = [f"<|{mode}|>"]
    h, w = grid.shape

    for r in range(h):
        tokens.append("<|start_row|>")
        cols = range(w) if (mode == "row_by_row" or r % 2 == 0) else range(w - 1, -1, -1)
        for c in cols:
            tokens.append(f"<|color_{int(grid[r, c])}|>")
        tokens.append("<|end_row|>")

    return tokens

def augment_task(task: Dict[str, Any], n_permutations: int = 3, seed: int = None) -> List[Dict[str, Any]]:
    augmented_examples = []
    base_seed = 0 if seed is None else seed

    for t_idx, t_id in enumerate(D4_TRANSFORMS):
        d4_task = {"id": f"{task['id']}_{t_id}", "train": [], "test": []}
        for inp, out in task["train"]:
            d4_task["train"].append((apply_d4(inp, t_id), apply_d4(out, t_id)))
        for inp, out in task["test"]:
            d4_inp = apply_d4(inp, t_id)
            d4_out = apply_d4(out, t_id) if out is not None else None
            d4_task["test"].append((d4_inp, d4_out))

        for p_idx in range(n_permutations):
            perm_seed = base_seed + (t_idx * 100) + p_idx
            perm_task = permute_colors(d4_task, seed=perm_seed)
            perm_task["id"] = f"{d4_task['id']}_perm{p_idx}"

            for mode in ["row_by_row", "snake"]:
                example = {
                    "task_id": perm_task["id"],
                    "mode": mode,
                    "train": [
                        (serialize(inp, mode=mode), serialize(out, mode=mode))
                        for inp, out in perm_task["train"]
                    ],
                    "test": [
                        (serialize(inp, mode=mode), serialize(out, mode=mode) if out is not None else None)
                        for inp, out in perm_task["test"]
                    ]
                }
                augmented_examples.append(example)

    return augmented_examples


def verify_holdout_isolation(build_dir: Path, holdout_dir: Path) -> bool:
    build_ids = {f.stem for f in Path(build_dir).glob("*.json")}
    holdout_ids = {f.stem for f in Path(holdout_dir).glob("*.json")}
    intersection = build_ids.intersection(holdout_ids)
    return len(intersection) == 0