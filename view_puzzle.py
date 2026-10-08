import sys
import json
from pathlib import Path
from src.core.grid import Grid
from src.visual_cot.renderer import save_grid_to_png

def inspect_task(task_path: str, save_png: bool = True):
    task_file = Path(task_path)
    if not task_file.exists():
        print(f"File not found: {task_path}")
        return

    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    task_id = task_file.stem
    print(f"\n================ TASK: {task_id} ================\n")

    # 1. Print and render Demonstration (Train) Pairs
    train_pairs = task_data.get("train", [])
    for idx, pair in enumerate(train_pairs):
        in_grid = Grid(pair["input"])
        out_grid = Grid(pair["output"])

        # Terminal ANSI Display
        in_grid.show(f"Train {idx + 1} - INPUT")
        out_grid.show(f"Train {idx + 1} - OUTPUT")

        # Save to PNG
        if save_png:
            out_dir = Path("rendered_puzzles") / task_id
            save_grid_to_png(in_grid, out_dir / f"train_{idx+1}_input.png")
            save_grid_to_png(out_grid, out_dir / f"train_{idx+1}_output.png")

    # 2. Print and render Test Pair(s)
    test_pairs = task_data.get("test", [])
    for idx, pair in enumerate(test_pairs):
        in_grid = Grid(pair["input"])
        in_grid.show(f"Test {idx + 1} - INPUT")

        if "output" in pair:  # Available in build/eval sets
            out_grid = Grid(pair["output"])
            out_grid.show(f"Test {idx + 1} - EXPECTED OUTPUT")
            if save_png:
                save_grid_to_png(out_grid, out_dir / f"test_{idx+1}_output.png")

        if save_png:
            save_grid_to_png(in_grid, out_dir / f"test_{idx+1}_input.png")

    if save_png:
        print(f"\n[Saved PNGs to rendered_puzzles/{task_id}/]")

if __name__ == "__main__":
    # Default to first puzzle in build if none provided
    sample_file = sys.argv[1] if len(sys.argv) > 1 else str(next(Path("data/build").glob("*.json")))
    inspect_task(sample_file)