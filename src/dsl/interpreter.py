"""
src/dsl/interpreter.py
Sequentially executes chains of DSL primitives on Grid objects with error boundaries.
"""

from typing import List, Dict, Any, Callable
from src.core.grid import Grid
from src.dsl import primitives


# Registry of atomic parameter-free or default-configured operations
PRIMITIVE_REGISTRY: Dict[str, Callable[[Grid], Grid]] = {
    "IDENTITY": primitives.identity,
    "ROT90": primitives.rotate_90,
    "ROT180": primitives.rotate_180,
    "ROT270": primitives.rotate_270,
    "FLIP_H": primitives.flip_h,
    "FLIP_V": primitives.flip_v,
    "CROP_CONTENT": primitives.crop_to_content,
    "PAD_BORDER": primitives.pad_to_border,
    "FILL_HOLES_YELLOW": lambda g: primitives.fill_enclosed_holes(g, fill_color=4),
    "FILL_HOLES_RED": lambda g: primitives.fill_enclosed_holes(g, fill_color=2),
    "KEEP_MODE_COLOR": primitives.keep_only_most_frequent_color,
    "SCALE_2X": lambda g: primitives.scale_by_factor(g, factor=2),
    "SCALE_3X": lambda g: primitives.scale_by_factor(g, factor=3),
}


class DSLInterpreter:
    @classmethod
    def execute(cls, program: List[str], initial_grid: Grid) -> Grid:
        """
        Executes a linear sequence of DSL operations:
        G_1 = op_1(G_0), G_2 = op_2(G_1), ...
        """
        current = initial_grid.copy()
        for op_name in program:
            if op_name not in PRIMITIVE_REGISTRY:
                raise ValueError(f"Unknown primitive '{op_name}' in program.")
            op_fn = PRIMITIVE_REGISTRY[op_name]
            current = op_fn(current)
        return current