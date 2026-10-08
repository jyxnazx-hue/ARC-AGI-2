import json
from pathlib import Path
from typing import List, Union
# pyrefly: ignore [missing-import]
import numpy as np

# ARC-AGI color specification mapped to ANSI 24-bit TrueColor RGB
ARC_ANSI_COLORS = {
    0: (0, 0, 0),        # 0: Black
    1: (0, 116, 217),    # 1: Blue
    2: (255, 65, 54),    # 2: Red
    3: (46, 204, 64),    # 3: Green
    4: (255, 220, 0),    # 4: Yellow
    5: (170, 170, 170),  # 5: Grey
    6: (240, 18, 190),   # 6: Magenta/Pink
    7: (255, 133, 27),   # 7: Orange
    8: (127, 219, 255),  # 8: Light Blue / Teal
    9: (135, 12, 37),    # 9: Maroon / Brown
}

RESET_CODE = "\033[0m"


class Grid:
    """Core 2D grid representation for ARC-AGI-2 tasks."""

    def __init__(self, data: Union[List[List[int]], np.ndarray]):
        self.cells = np.array(data, dtype=np.int8)
        if self.cells.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {self.cells.shape}")
        self.height, self.width = self.cells.shape

    @classmethod
    def from_json(cls, json_path: Union[str, Path], split: str = "train", index: int = 0, pair: str = "input") -> "Grid":
        """
        Loads a Grid instance from an ARC task JSON file.
        split: 'train' or 'test'
        index: demonstration pair index
        pair: 'input' or 'output'
        """
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            task = json.load(f)
        matrix = task[split][index][pair]
        return cls(matrix)

    def to_ansi(self, show_numbers: bool = False) -> str:
        """
        Renders the grid to an ANSI TrueColor terminal string.
        Each cell is rendered as two characters wide for square pixel aspect ratio.
        """
        lines = []
        for row in self.cells:
            line_parts = []
            for val in row:
                r, g, b = ARC_ANSI_COLORS.get(int(val), (255, 255, 255))
                bg = f"\033[48;2;{r};{g};{b}m"
                if show_numbers:
                    fg = "\033[38;2;0;0;0m" if val in (4, 5, 8) else "\033[38;2;255;255;255m"
                    line_parts.append(f"{bg}{fg}{val:2d}{RESET_CODE}")
                else:
                    line_parts.append(f"{bg}  {RESET_CODE}")
            lines.append("".join(line_parts))
        return "\n".join(lines)

    def show(self, title: str = None, show_numbers: bool = False) -> None:
        """Prints the ANSI representation directly to standard output."""
        if title:
            print(f"=== {title} ({self.height}x{self.width}) ===")
        else:
            print(f"Grid ({self.height}x{self.width}):")
        print(self.to_ansi(show_numbers=show_numbers))

    # Dihedral Group D4 Transformations (8 orthogonal symmetries)
    def rot90(self, k: int = 1) -> "Grid":
        """Rotate clockwise/counterclockwise by 90-degree increments."""
        return Grid(np.rot90(self.cells, k=k))

    def flip_horizontal(self) -> "Grid":
        """Reflect left-to-right along vertical axis."""
        return Grid(np.fliplr(self.cells))

    def flip_vertical(self) -> "Grid":
        """Reflect top-to-bottom along horizontal axis."""
        return Grid(np.flipud(self.cells))

    def get_d4_symmetries(self) -> List["Grid"]:
        """Returns all 8 unique spatial transformations in the D4 group."""
        symmetries = []
        for k in range(4):
            rotated = np.rot90(self.cells, k=k)
            symmetries.append(Grid(rotated))
            symmetries.append(Grid(np.fliplr(rotated)))
        return symmetries

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.cells, other.cells)

    def __repr__(self) -> str:
        return f"Grid({self.height}x{self.width})\n" + "\n".join(
            "".join(str(val) for val in row) for row in self.cells
        )