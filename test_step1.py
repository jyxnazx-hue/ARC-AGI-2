import json
from pathlib import Path
from src.core.grid import Grid
from src.core.invariance import generate_d4_transformations, augment_demonstration_pair

def test_pipeline():
    task_file = next(Path("data/build").glob("*.json"))
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    demo_in = Grid(data["train"][0]["input"])
    demo_out = Grid(data["train"][0]["output"])

    print(f"Original Demo Input shape: {demo_in.shape}")
    augmented = augment_demonstration_pair(demo_in, demo_out)
    
    print(f"Generated {len(augmented)} invariant pairs successfully!")
    assert len(augmented) == 8, "D4 must generate exactly 8 transformed pairs."
    
    # Visual check on the terminal
    demo_in.show("Original Input")
    augmented[1][0].show("Rotated 90 Deg View")
    print("Step 1 validation PASSED.")

if __name__ == "__main__":
    test_pipeline()