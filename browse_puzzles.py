import random
import sys
from pathlib import Path
from src.core.grid import Grid

def load_catalog(build_count=40, eval_count=10, seed=42):
    random.seed(seed)
    build_files = sorted(list(Path("data/build").glob("*.json")))
    eval_files = sorted(list(Path("data/eval_public").glob("*.json")))

    selected_build = random.sample(build_files, min(build_count, len(build_files)))
    selected_eval = random.sample(eval_files, min(eval_count, len(eval_files)))

    catalog = [("BUILD", p) for p in selected_build] + [("EVAL", p) for p in selected_eval]
    return catalog

def render_task(split_type, task_path):
    import json
    with open(task_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id = task_path.stem
    print("\n" + "=" * 50)
    print(f"[{split_type}] Task ID: {task_id}")
    print("=" * 50)

    train_pairs = data.get("train", [])
    print(f"\n--- Demonstrations ({len(train_pairs)} pairs) ---")
    for idx, pair in enumerate(train_pairs):
        g_in = Grid(pair["input"])
        g_out = Grid(pair["output"])
        g_in.show(f"Demo {idx+1} INPUT")
        g_out.show(f"Demo {idx+1} OUTPUT")

    test_pairs = data.get("test", [])
    print(f"\n--- Test Input ({len(test_pairs)} challenge) ---")
    for idx, pair in enumerate(test_pairs):
        g_in = Grid(pair["input"])
        g_in.show(f"Test {idx+1} INPUT")
        if "output" in pair:
            g_out = Grid(pair["output"])
            g_out.show(f"Test {idx+1} EXPECTED OUTPUT")

def main():
    catalog = load_catalog()
    if not catalog:
        print("No tasks found in data/build or data/eval_public.")
        return

    idx = 0
    total = len(catalog)

    while True:
        split_type, task_path = catalog[idx]
        print(f"\nViewing {idx + 1} of {total}: {task_path.name}")
        render_task(split_type, task_path)

        cmd = input("\n[n]ext, [p]revious, [q]uit, or enter index (1-50): ").strip().lower()
        if cmd == "q":
            break
        elif cmd in ("n", ""):
            if idx < total - 1:
                idx += 1
            else:
                print("Reached the end of the 50 selected tasks.")
        elif cmd == "p":
            if idx > 0:
                idx -= 1
        elif cmd.isdigit():
            target = int(cmd) - 1
            if 0 <= target < total:
                idx = target

if __name__ == "__main__":
    main()