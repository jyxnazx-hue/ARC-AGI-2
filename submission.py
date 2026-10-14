import json
import os
from pathlib import Path
from typing import Dict, Any
from src.search.brute_force import BruteForceSearcher


def locate_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Finds and loads ARC tasks across Kaggle 2026 competition inputs and local paths.
    Returns: Dict mapping task_id -> task_data dict {"train": [...], "test": [...]}
    """
    tasks = {}

    candidate_sources = [
        Path("/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"),
        Path("/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"),
        Path("/kaggle/input/arc-prize-2026-arc-agi-2/test"),
        Path("data/eval_public"),
        Path("data/build"),
    ]

    selected_source = None
    for src in candidate_sources:
        if src.exists():
            selected_source = src
            break

    if selected_source is None:
        raise FileNotFoundError("Could not locate any valid ARC-AGI-2 data source.")

    print(f"Loading tasks from: {selected_source}")

    if selected_source.is_file() and selected_source.suffix == ".json":
        with open(selected_source, "r", encoding="utf-8") as f:
            tasks = json.load(f)

    elif selected_source.is_dir():
        for fpath in sorted(selected_source.glob("*.json")):
            with open(fpath, "r", encoding="utf-8") as f:
                tasks[fpath.stem] = json.load(f)

    return tasks


def run_submission(output_file: Path = Path("submission.json")):
    searcher = BruteForceSearcher(max_depth=2)
    tasks = locate_tasks()
    print(f"Loaded {len(tasks)} tasks. Starting solver...")

    submission = {}
    for idx, (task_id, task_data) in enumerate(tasks.items(), 1):
        try:
            preds = searcher.solve_task(task_data)
            submission[task_id] = preds
        except Exception:
            test_cases = task_data.get("test", [])
            fallback_attempts = []
            for t in test_cases:
                inp = t.get("input", [[0]])
                fallback_attempts.append({"attempt_1": inp, "attempt_2": inp})
            submission[task_id] = fallback_attempts

        if idx % 20 == 0 or idx == len(tasks):
            print(f"Processed [{idx}/{len(tasks)}] tasks.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    print(f"\nDone! Submission saved to {output_file.resolve()} ({len(submission)} tasks).")


if __name__ == "__main__":
    run_submission()