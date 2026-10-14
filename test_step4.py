# pyrefly: ignore [missing-import]
import numpy as np
from src.core.grid import Grid
from src.dsl.interpreter import DSLInterpreter

def test_dsl():
    # 1. Test cavity fill (Topology)
    # A 5x5 grid with an enclosed loop of green (3)
    loop_mat = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)
    grid = Grid(loop_mat)

    # Execute chain: Fill enclosed holes with yellow (4), then Crop to content
    program = ["FILL_HOLES_YELLOW", "CROP_CONTENT"]
    result = DSLInterpreter.execute(program, grid)

    assert result.shape == (3, 3), f"Expected (3, 3), got {result.shape}"
    assert result.matrix[1, 1] == 4, "Trapped cavity must be filled with yellow (4)."
    print(f"Chained output shape: {result.shape}, Center pixel: {result.matrix[1, 1]}")

    # 2. Test spatial rotation chain
    rot_prog = ["ROT90", "ROT180"]
    rot_res = DSLInterpreter.execute(rot_prog, grid)
    # ROT90 + ROT180 = ROT270
    expected = grid.rotate(3)
    assert rot_res == expected, "Interpreter composition must match exact transformation."

    print("Step 4 Relative DSL & Interpreter validation PASSED.")

if __name__ == "__main__":
    test_dsl()