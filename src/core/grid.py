"""
src/core/grid.py
Object-oriented wrapper around 2D NumPy matrices for ARC-AGI-2.
Enforces integer types, boundary safety, and relative transformations.
"""

from typing import Tuple, List, Union, Dict
# pyrefly: ignore [missing-import]
import numpy as np


class Grid:
    def __init__(self, data: Union[List[List[int]], np.ndarray]):
        if isinstance(data, np.ndarray):
            self.matrix = data.astype(np.int8)
        else:
            self.matrix = np.array(data, dtype=np.int8)
            
        if self.matrix.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {self.matrix.shape}")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape

    @property
    def height(self) -> int:
        return self.matrix.shape[0]

    @property
    def width(self) -> int:
        return self.matrix.shape[1]

    def to_list(self) -> List[List[int]]:
        return self.matrix.tolist()

    def copy(self) -> "Grid":
        return Grid(self.matrix.copy())

    def rotate(self, k: int = 1) -> "Grid":
        """Rotates grid by 90 degrees * k counter-clockwise (or clockwise if k=-1)."""
        return Grid(np.rot90(self.matrix, k=k))

    def flip_horizontal(self) -> "Grid":
        """Flips grid left-to-right across vertical axis."""
        return Grid(np.fliplr(self.matrix))

    def flip_vertical(self) -> "Grid":
        """Flips grid upside-down across horizontal axis."""
        return Grid(np.flipud(self.matrix))

    def crop(self, min_r: int, min_c: int, max_r: int, max_c: int) -> "Grid":
        """Safely crops a subgrid within bounding box limits (inclusive)."""
        min_r = max(0, min_r)
        min_c = max(0, min_c)
        max_r = min(self.height - 1, max_r)
        max_c = min(self.width - 1, max_c)
        return Grid(self.matrix[min_r : max_r + 1, min_c : max_c + 1])

    def recolor(self, mapping: Dict[int, int]) -> "Grid":
        """Substitutes color values according to a dictionary."""
        new_mat = self.matrix.copy()
        for old_c, new_c in mapping.items():
            new_mat[self.matrix == old_c] = new_c
        return Grid(new_mat)

    def show(self, title: str = ""):
        """ANSI color terminal display."""
        colors = {
            0: "\033[40m  \033[0m",  # Black
            1: "\033[44m  \033[0m",  # Blue
            2: "\033[41m  \033[0m",  # Red
            3: "\033[42m  \033[0m",  # Green
            4: "\033[43m  \033[0m",  # Yellow
            5: "\033[47m  \033[0m",  # Grey
            6: "\033[45m  \033[0m",  # Pink/Magenta
            7: "\033[48;5;208m  \033[0m",  # Orange
            8: "\033[46m  \033[0m",  # Sky Blue / Cyan
            9: "\033[48;5;88m  \033[0m",   # Maroon
        }
        if title:
            print(f"--- {title} ({self.height}x{self.width}) ---")
        for row in self.matrix:
            print("".join(colors.get(int(c), "  ") for c in row))
        print()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.matrix, other.matrix)