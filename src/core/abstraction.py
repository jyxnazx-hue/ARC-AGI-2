"""
src/core/abstraction.py
GPAR Dual-Abstraction: Extracts discrete connected components under
both 4-way orthogonal (CC4) and 8-way diagonal (CC8) neighborhood structures.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.ndimage import label
from src.core.grid import Grid


@dataclass
class ConnectedComponent:
    label_id: int
    color: int
    area: int
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    coords: np.ndarray               # Array of (r, c) coordinates
    mask: np.ndarray                 # Boolean mask matching canvas shape

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.label_id,
            "color": self.color,
            "area": self.area,
            "bbox": self.bbox,
            "dimensions": (self.height, self.width),
        }


class GPARAbstraction:
    # 4-connected structuring element (orthogonal edges only)
    KERNEL_4 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=int)

    # 8-connected structuring element (orthogonal edges + diagonal corners)
    KERNEL_8 = np.ones((3, 3), dtype=int)

    @classmethod
    def get_kernel(cls, mode: str) -> np.ndarray:
        mode_clean = mode.lower().replace("-", "").replace("_", "")
        if "4" in mode_clean:
            return cls.KERNEL_4
        elif "8" in mode_clean:
            return cls.KERNEL_8
        raise ValueError(f"Unknown connectivity mode '{mode}'. Choose '4-conn' or '8-conn'.")

    @classmethod
    def extract_components(
        cls,
        grid: Grid,
        connectivity: str = "4-conn",
        ignore_color: Optional[int] = 0
    ) -> List[ConnectedComponent]:
        """
        Segments grid into discrete ConnectedComponent objects based on color
        uniformity and spatial connectivity.
        """
        kernel = cls.get_kernel(connectivity)
        mat = grid.matrix
        unique_colors = np.unique(mat)
        components: List[ConnectedComponent] = []
        comp_id = 1

        for c in unique_colors:
            if ignore_color is not None and c == ignore_color:
                continue

            color_mask = (mat == c).astype(int)
            labeled_arr, num_features = label(color_mask, structure=kernel)

            for feat_idx in range(1, num_features + 1):
                feat_mask = (labeled_arr == feat_idx)
                coords = np.argwhere(feat_mask)

                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)

                components.append(ConnectedComponent(
                    label_id=comp_id,
                    color=int(c),
                    area=len(coords),
                    bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
                    coords=coords,
                    mask=feat_mask
                ))
                comp_id += 1

        return components

    @classmethod
    def dual_abstract(
        cls,
        grid: Grid,
        ignore_color: Optional[int] = 0
    ) -> Dict[str, List[ConnectedComponent]]:
        """
        Returns both CC4 and CC8 abstractions simultaneously for GPAR switching.
        """
        return {
            "4-conn": cls.extract_components(grid, connectivity="4-conn", ignore_color=ignore_color),
            "8-conn": cls.extract_components(grid, connectivity="8-conn", ignore_color=ignore_color)
        }