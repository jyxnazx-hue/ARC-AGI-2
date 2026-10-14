import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import load_task, augment_task, verify_holdout_isolation


def test_components_5_and_6():
    build_dir = Path("data/build")
    holdout_dir = Path("data/holdout")

    # 1. Component 6: Verify 0 Holdout Leakage
    assert verify_holdout_isolation(build_dir, holdout_dir) is True, "Holdout data leakage detected"
    print("HOLDOUT ISOLATION VERIFIED (0 Leakage)")

    # 2. Component 5: Augment 1 task and verify volume
    # 8 D4 transforms * 4 permutations * 2 serializations = 64 examples/task
    first_task = load_task(next(build_dir.glob("*.json")))
    examples = augment_task(first_task, n_permutations=4, seed=42)
    assert len(examples) == 64
    assert len(examples[0]["train"]) == len(first_task["train"])
    assert examples[0]["train"][0][0][0] in ["<|row_by_row|>", "<|snake|>"]
    print(f"TASK AUGMENTATION VERIFIED ({len(examples)} variations generated per task)")

    # 3. Projected size check across all 800 tasks
    total_tasks = len(list(build_dir.glob("*.json")))
    projected_size = total_tasks * len(examples)
    print(f"Projected Dataset Size: {projected_size} examples (Target: >= 40,000)")
    assert projected_size >= 40000


if __name__ == "__main__":
    test_components_5_and_6()