import random
import sys
from pathlib import Path
from src.visual_cot.renderer import render_full_puzzle

def generate_phase0_catalog(build_count=40, eval_count=10, seed=42):
    random.seed(seed)
    
    build_dir = Path("data/build")
    eval_dir = Path("data/eval_public")
    
    build_files = sorted(list(build_dir.glob("*.json")))
    eval_files = sorted(list(eval_dir.glob("*.json")))
    
    if len(build_files) < build_count:
        print(f"Warning: only found {len(build_files)} files in {build_dir}")
        build_count = len(build_files)
        
    if len(eval_files) < eval_count:
        print(f"Warning: only found {len(eval_files)} files in {eval_dir}")
        eval_count = len(eval_files)
        
    selected_build = random.sample(build_files, build_count)
    selected_eval = random.sample(eval_files, eval_count)
    
    return [("BUILD", p) for p in selected_build] + [("EVAL", p) for p in selected_eval]

def main():
    catalog = generate_phase0_catalog(40, 10)
    out_dir = Path("rendered_sheets")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting render of {len(catalog)} full-puzzle sheets into '{out_dir}/'...\n")
    
    for i, (split_name, task_path) in enumerate(catalog, start=1):
        target_name = f"{i:02d}_{split_name}_{task_path.stem}.png"
        target_dest = out_dir / target_name
        
        try:
            render_full_puzzle(task_path, output_path=target_dest, cell_size=18)
            print(f"[{i:02d}/50] Rendered: {target_name}")
        except Exception as e:
            print(f"[{i:02d}/50] Error rendering {task_path.name}: {e}")

    print(f"\nAll 50 puzzle sheets are ready in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()