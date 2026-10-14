# pyrefly: ignore [missing-import]
import numpy as np
from src.core.grid import Grid
from src.core.guardrails import check_dimension_bounds, check_color_consistency

def test_guardrails():
    t_in = Grid(np.array([[1, 2], [0, 0]]))
    t_out = Grid(np.array([[2, 1], [0, 0]]))
    test_in = Grid(np.array([[1, 0], [2, 0]]))
    train_pairs = [(t_in, t_out)]

    # 1. Valid candidate (only uses colors 0, 1, 2)
    valid_cand = Grid(np.array([[2, 0], [1, 0]]))
    assert check_dimension_bounds(valid_cand) is True
    assert check_color_consistency(valid_cand, test_in, train_pairs) is True

    # 2. Invalid candidate: hallucinated color 7 (Orange)
    bad_color_cand = Grid(np.array([[7, 0], [1, 0]]))
    assert check_color_consistency(bad_color_cand, test_in, train_pairs) is False, \
        "Guardrail must catch hallucinated color 7."

    # 3. Invalid candidate: out of bounds (31x2)
    bad_dim_cand = Grid(np.zeros((31, 2), dtype=int))
    assert check_dimension_bounds(bad_dim_cand) is False, \
        "Guardrail must catch dimensions > 30."

    print("Step 3 Guardrails validation PASSED.")

if __name__ == "__main__":
    test_guardrails()