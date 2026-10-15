import numpy as np
from typing import Tuple

class ARCTokenizer:
    PAD_TOKEN = 0
    EOS_TOKEN = 1
    COLOR_OFFSET = 2
    VOCAB_SIZE = 12
    CANVAS_SIZE = 30
    CANVAS_TOKENS = 900

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

    def grid_to_canvas(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = grid.shape
        canvas_2d = np.full((self.CANVAS_SIZE, self.CANVAS_SIZE), self.PAD_TOKEN, dtype=np.int64)
        loss_mask_2d = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE), dtype=np.float32)

        canvas_2d[:h, :w] = grid + self.COLOR_OFFSET
        loss_mask_2d[:h, :w] = 1.0

        if h < self.CANVAS_SIZE:
            canvas_2d[h, :w] = self.EOS_TOKEN
        if w < self.CANVAS_SIZE:
            canvas_2d[:h, w] = self.EOS_TOKEN

        return canvas_2d.flatten(), loss_mask_2d.flatten()

    def canvas_to_grid(self, canvas: np.ndarray, h: int = None, w: int = None) -> np.ndarray:
        arr = np.array(canvas, dtype=np.int64).reshape(self.CANVAS_SIZE, self.CANVAS_SIZE)
        if h is not None and w is not None:
            grid_tokens = arr[:h, :w]
            return np.clip(grid_tokens - self.COLOR_OFFSET, 0, 9)

        valid = (arr >= self.COLOR_OFFSET) & (arr <= 11)
        rows, cols = np.where(valid)
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros((1, 1), dtype=np.int64)

        max_r, max_c = rows.max() + 1, cols.max() + 1
        return np.clip(arr[:max_r, :max_c] - self.COLOR_OFFSET, 0, 9)
