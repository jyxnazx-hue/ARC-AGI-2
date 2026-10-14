"""
src/dsl/primitives.py
Relative transformation library for ARC-AGI-2.
Enforces the Relative Rule: all operations are invariant to absolute coordinate positions.
"""

from typing import Optional, Dict
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.ndimage import binary_fill_holes
from src.core.grid import Grid
from src.core.abstraction import GPARAbstraction


def crop_to_content(grid: Grid, bg: int = 0) -> Grid:
    """Crops the canvas tightly around all non-background pixels."""
    coords = np.argwhere(grid.matrix != bg)
    if coords.size == 0:
        return grid.copy()
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    return grid.crop(min_r, min_c, max_r, max_c)


def pad_to_border(grid: Grid, pad_width: int = 1, fill_color: int = 0) -> Grid:
    """Pads the grid perimeter uniformly with a specified color."""
    padded = np.pad(grid.matrix, pad_width, mode="constant", constant_values=fill_color)
    return Grid(padded)


def rotate_90(grid: Grid) -> Grid:
    """Rotates grid 90 degrees counter-clockwise."""
    return grid.rotate(k=1)


def rotate_180(grid: Grid) -> Grid:
    """Rotates grid 180 degrees."""
    return grid.rotate(k=2)


def rotate_270(grid: Grid) -> Grid:
    """Rotates grid 270 degrees counter-clockwise."""
    return grid.rotate(k=3)


def flip_h(grid: Grid) -> Grid:
    """Reflects grid horizontally (left-to-right)."""
    return grid.flip_horizontal()


def flip_v(grid: Grid) -> Grid:
    """Reflects grid vertically (up-to-down)."""
    return grid.flip_vertical()


def fill_enclosed_holes(grid: Grid, fill_color: int = 4, bg: int = 0) -> Grid:
    """
    Topology Prior: Fills background cavities trapped inside 8-connected walls.
    Exterior background remains unchanged.
    """
    wall_mask = (grid.matrix != bg)
    filled_mask = binary_fill_holes(wall_mask)
    hole_mask = filled_mask & (~wall_mask)

    result = grid.matrix.copy()
    result[hole_mask] = fill_color
    return Grid(result)


def recolor_largest_object(
    grid: Grid,
    target_color: int,
    connectivity: str = "4-conn",
    bg: int = 0
) -> Grid:
    """Numerosity Prior: Finds the connected component with maximal area and recolors it."""
    objs = GPARAbstraction.extract_components(grid, connectivity=connectivity, ignore_color=bg)
    if not objs:
        return grid.copy()
    largest = max(objs, key=lambda o: o.area)
    result = grid.matrix.copy()
    result[largest.mask] = target_color
    return Grid(result)


def recolor_smallest_object(
    grid: Grid,
    target_color: int,
    connectivity: str = "4-conn",
    bg: int = 0
) -> Grid:
    """Numerosity Prior: Finds the connected component with minimal area and recolors it."""
    objs = GPARAbstraction.extract_components(grid, connectivity=connectivity, ignore_color=bg)
    if not objs:
        return grid.copy()
    smallest = min(objs, key=lambda o: o.area)
    result = grid.matrix.copy()
    result[smallest.mask] = target_color
    return Grid(result)


def keep_only_most_frequent_color(grid: Grid, bg: int = 0) -> Grid:
    """Keeps only pixels belonging to the mode non-background color; clears rest to bg."""
    mat = grid.matrix
    non_bg = mat[mat != bg]
    if non_bg.size == 0:
        return grid.copy()
    values, counts = np.unique(non_bg, return_counts=True)
    mode_color = values[np.argmax(counts)]

    result = np.full_like(mat, bg)
    result[mat == mode_color] = mode_color
    return Grid(result)


def scale_by_factor(grid: Grid, factor: int = 2) -> Grid:
    """Upscales grid by repeating each pixel as an (factor x factor) block."""
    if factor <= 1:
        return grid.copy()
    scaled = np.repeat(np.repeat(grid.matrix, factor, axis=0), factor, axis=1)
    return Grid(scaled)


def extract_center_crop(grid: Grid, crop_h: int = 3, crop_w: int = 3) -> Grid:
    """Extracts a relative central window from the canvas."""
    h, w = grid.shape
    if crop_h > h or crop_w > w:
        return grid.copy()
    start_r = (h - crop_h) // 2
    start_c = (w - crop_w) // 2
    return grid.crop(start_r, start_c, start_r + crop_h - 1, start_c + crop_w - 1)


def replace_color(grid: Grid, source_color: int, target_color: int) -> Grid:
    """Substitutes one color with another across the entire grid."""
    return grid.recolor({source_color: target_color})


def identity(grid: Grid) -> Grid:
    """Returns an identical copy of the grid."""
    return grid.copy()