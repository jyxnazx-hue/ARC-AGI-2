"""
src/core/guardrails.py
White-box filtering priors to prune invalid candidate grids before verification.
"""

from typing import List, Tuple, Set, Optional
# pyrefly: ignore [missing-import]
import numpy as np
from src.core.grid import Grid


def get_palette(grid_or_mat) -> Set[int]:
    """Returns the set of unique integer colors present in a Grid or ndarray."""
    mat = grid_or_mat.matrix if isinstance(grid_or_mat, Grid) else grid_or_mat
    return set(np.unique(mat).tolist())


def check_dimension_bounds(
    candidate: Grid,
    expected_shape: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Verifies candidate dimensions fall within valid ARC boundaries (1x1 to 30x30).
    Optionally checks if candidate matches an expected shape.
    """
    mat = candidate.matrix if isinstance(candidate, Grid) else candidate
    if mat.ndim != 2:
        return False
    h, w = mat.shape
    if not (1 <= h <= 30 and 1 <= w <= 30):
        return False
    if expected_shape is not None and (h, w) != expected_shape:
        return False
    return True


def check_color_consistency(
    candidate: Grid,
    test_input: Grid,
    train_pairs: List[Tuple[Grid, Grid]]
) -> bool:
    """
    Validates whether the candidate output respects the color conservation filter:
    C(O_pred) ⊆ ( U C(I_train) U C(O_train) ) U C(I_test)
    """
    allowed_colors = get_palette(test_input)
    for inp, out in train_pairs:
        allowed_colors.update(get_palette(inp))
        allowed_colors.update(get_palette(out))

    cand_colors = get_palette(candidate)
    return cand_colors.issubset(allowed_colors)