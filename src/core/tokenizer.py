"""
Fixed 30x30 canvas tokenizer for ARC-AGI-2

Vocabulary (12 tokens):
    0  <pad>    filler for positions outside the real grid
    1  <mask>   denoising mask (used by inference / DRM)
    2  <c0>     color 0  (black / background in ARC)
    3  <c1>
    ...
    11 <c9>

Every example is encoded onto a fixed (30, 30) canvas, placed at top-left (0,0),
and padded to 900 tokens. Decode extracts the bounding box of non-pad cells.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# pyrefly: ignore[missing-import]
import numpy as np

# Constants
CANVAS_H: int = 30
CANVAS_W: int = 30
CANVAS_SIZE: int = CANVAS_H * CANVAS_W  

PAD: int = 0
MASK: int = 1
COLOR_OFFSET: int = 2  # color c -> token c + COLOR_OFFSET
NUM_COLORS: int = 10
VOCAB_SIZE: int = 12

COLOR_TOKEN_IDS: tuple[int, ...] = tuple(range(COLOR_OFFSET, COLOR_OFFSET + NUM_COLORS))

# Core encode / decode
def encode_grid(
    grid: Sequence[Sequence[int]] | np.ndarray,
    canvas: np.ndarray | None = None,
    offset_h: int = 0,
    offset_w: int = 0,
) -> np.ndarray:
    """
    Place `grid` at (offset_h, offset_w) on a fixed 30x30 canvas.

    Cells outside the placed grid are left as PAD. If `canvas` is provided,
    it is modified in place (caller is responsible for that); otherwise a new
    all-PAD canvas is allocated.

    Returns
    -------
    np.ndarray of shape (30, 30), dtype int64, values in [0, 11].
    """
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {g.shape}")
    h, w = g.shape
    if h == 0 or w == 0:
        raise ValueError("grid must be non-empty")
    if h > CANVAS_H or w > CANVAS_W:
        raise ValueError(f"grid {h}x{w} exceeds canvas {CANVAS_H}x{CANVAS_W}")
    if offset_h < 0 or offset_w < 0:
        raise ValueError("offsets must be non-negative")
    if offset_h + h > CANVAS_H or offset_w + w > CANVAS_W:
        raise ValueError(
            f"grid {h}x{w} at offset ({offset_h},{offset_w}) does not fit "
            f"on {CANVAS_H}x{CANVAS_W} canvas"
        )
    if not np.all((g >= 0) & (g < NUM_COLORS)):
        raise ValueError(f"grid values must be in [0, {NUM_COLORS - 1}]")

    if canvas is None:
        canvas = np.full((CANVAS_H, CANVAS_W), PAD, dtype=np.int64)
    else:
        if canvas.shape != (CANVAS_H, CANVAS_W):
            raise ValueError(
                f"canvas must be {CANVAS_H}x{CANVAS_W}, got {canvas.shape}"
            )

    canvas[offset_h : offset_h + h, offset_w : offset_w + w] = g + COLOR_OFFSET
    return canvas


def decode_canvas(canvas: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """
    Extract the bounding box of non-PAD cells and undo the color offset.

    Raises if any MASK token is present — you cannot decode a partially-masked
    canvas to a real grid. Callers doing inference should argmax to a color
    token first (see `argmax_to_colors`).
    """
    c = np.asarray(canvas, dtype=np.int64)
    if c.shape != (CANVAS_H, CANVAS_W):
        raise ValueError(f"canvas must be {CANVAS_H}x{CANVAS_W}, got {c.shape}")

    if (c == MASK).any():
        raise ValueError(
            "canvas contains MASK tokens; resolve to color tokens before decoding"
        )

    non_pad = c != PAD
    if not non_pad.any():
        return np.zeros((0, 0), dtype=np.int64)

    rows = np.any(non_pad, axis=1)
    cols = np.any(non_pad, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    box = c[r0 : r1 + 1, c0 : c1 + 1]
    return box - COLOR_OFFSET

# Flatten / unflatten (for feeding the model)
def flatten(canvas: np.ndarray) -> np.ndarray:
    """(30, 30) -> (900,) int64."""
    return np.asarray(canvas, dtype=np.int64).reshape(-1)


def unflatten(flat: Sequence[int] | np.ndarray) -> np.ndarray:
    """(900,) -> (30, 30) int64."""
    arr = np.asarray(flat, dtype=np.int64)
    if arr.shape != (CANVAS_SIZE,):
        raise ValueError(f"expected shape ({CANVAS_SIZE},), got {arr.shape}")
    return arr.reshape(CANVAS_H, CANVAS_W)


# Helpers used by loss / inference
def color_logit_mask() -> np.ndarray:
    """
    Boolean mask of shape (VOCAB_SIZE,) that is True for color tokens only.
    Use at inference: logits[..., ~mask] = -inf so argmax can never pick
    PAD or MASK.
    """
    m = np.zeros(VOCAB_SIZE, dtype=bool)
    m[list(COLOR_TOKEN_IDS)] = True
    return m


def argmax_to_colors(logits: np.ndarray) -> np.ndarray:
    """
    logits: (..., VOCAB_SIZE) -> token ids in [2, 11].
    Restricts argmax to color tokens so PAD/MASK can never win.
    """
    logits = np.asarray(logits)
    if logits.shape[-1] != VOCAB_SIZE:
        raise ValueError(f"last dim must be {VOCAB_SIZE}, got {logits.shape[-1]}")
    color_logits = logits[..., list(COLOR_TOKEN_IDS)]
    idx = np.argmax(color_logits, axis=-1)
    return idx + COLOR_OFFSET


def pad_mask(canvas: np.ndarray) -> np.ndarray:
    """
    (30, 30) -> (30, 30) bool, True where the cell is NOT pad.
    Useful for building the loss mask downstream.
    """
    return np.asarray(canvas, dtype=np.int64) != PAD