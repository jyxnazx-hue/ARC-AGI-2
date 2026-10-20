# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import pytest
from src.corrupt import corrupt_canvas, mask_ratio

def test_mask_ratio_math():
    assert mask_ratio(0.0) == 0.0
    assert mask_ratio(1.0) == 1.0
    assert 0.49 < mask_ratio(0.5) < 0.51

def test_corrupt_canvas():
    canvas = torch.tensor([2, 3, 4, 5, 0, 0])
    mask = torch.tensor([1, 1, 1, 1, 0, 0])

    # Tau = 0 (No corruption)
    out0 = corrupt_canvas(canvas, mask, tau=0.0)
    assert torch.equal(out0, canvas)

    # Tau = 1 
    out1 = corrupt_canvas(canvas, mask, tau=1.0)
    assert out1[0] == 1 and out1[1] == 1 and out1[2] == 1 and out1[3] == 1
    assert out1[4] == 0 and out1[5] == 0 

    # Tau = 0.5 (Half corrupted)
    out5 = corrupt_canvas(canvas, mask, tau=0.5)
    n_masked = (out5[:4] == 1).sum().item()
    assert n_masked == 2 