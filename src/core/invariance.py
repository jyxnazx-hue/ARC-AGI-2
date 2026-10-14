"""
src/core/invariance.py
Implementation of Dihedral Group D4 transformations (8 spatial symmetries).
Used for data augmentation and multi-perspective consensus.
"""

from typing import List, Tuple
from src.core.grid import Grid


def generate_d4_transformations(grid: Grid) -> List[Grid]:
    """
    Produces the complete set of 8 D4 dihedral symmetries:
    1. Identity (0°)
    2. Rotation 90°
    3. Rotation 180°
    4. Rotation 270°
    5. Flip Horizontal
    6. Flip Vertical
    7. Transpose (Main Diagonal)
    8. Anti-Transpose (Anti-Diagonal)
    """
    transforms = []
    # 4 Rotations
    for k in range(4):
        transforms.append(grid.rotate(k=k))
        
    # Flip horizontally, then rotate
    flipped = grid.flip_horizontal()
    for k in range(4):
        transforms.append(flipped.rotate(k=k))
        
    return transforms


def augment_demonstration_pair(in_grid: Grid, out_grid: Grid) -> List[Tuple[Grid, Grid]]:
    """
    Transforms both input and output grids symmetrically, preserving the logical rule.
    Returns 8 valid (in, out) demonstration pairs from 1 original pair.
    """
    in_views = generate_d4_transformations(in_grid)
    out_views = generate_d4_transformations(out_grid)
    return list(zip(in_views, out_views))