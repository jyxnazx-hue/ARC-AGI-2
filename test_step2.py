# pyrefly: ignore [missing-import]
import numpy as np
from src.core.grid import Grid
from src.core.abstraction import GPARAbstraction

def test_dual_abstraction():
    # Diagnostic diagonal pattern:
    # [1, 0]
    # [0, 1]
    diag_pattern = np.array([
        [1, 0],
        [0, 1]
    ], dtype=int)

    grid = Grid(diag_pattern)

    # 1. Test 4-connectivity: Diagonals must be 2 SEPARATE objects
    cc4 = GPARAbstraction.extract_components(grid, connectivity="4-conn", ignore_color=0)
    print(f"4-conn detected {len(cc4)} object(s). (Expected: 2)")
    assert len(cc4) == 2, "4-conn must separate diagonal pixels."

    # 2. Test 8-connectivity: Diagonals must merge into 1 SINGLE object
    cc8 = GPARAbstraction.extract_components(grid, connectivity="8-conn", ignore_color=0)
    print(f"8-conn detected {len(cc8)} object(s). (Expected: 1)")
    assert len(cc8) == 1, "8-conn must join diagonal pixels."

    # 3. Test dual_abstract dictionary mapping
    dual = GPARAbstraction.dual_abstract(grid, ignore_color=0)
    assert "4-conn" in dual and "8-conn" in dual

    print("Step 2 GPAR Dual-Abstraction validation PASSED.")

if __name__ == "__main__":
    test_dual_abstraction()